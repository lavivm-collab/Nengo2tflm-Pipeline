import os
import tempfile
import shutil
from collections import deque
from typing import List, Union, Dict, Deque, TypeVar
import nengo
import tensorflow as tf

# A component of the Nengo DAG we compile: either a population of neurons or an
# input/output port. Used throughout instead of repeating the Union inline, which had
# already drifted inconsistent (some signatures wrote it as Union[Node, Ensemble] instead).
NengoGraphObject = Union[nengo.Ensemble, nengo.Node]


# =====================================================================
# STAGE 1: GRAPH TOPOLOGY & LOOP VALIDATION
# =====================================================================

def topological_sort_and_detect_loops(network: nengo.Network) -> List[NengoGraphObject]:
    """
    Performs a topological sort on Nengo network components using Kahn's Algorithm.
    Protects the deployment target by aborting if an un-routable recurrent cycle is detected.
    """
    all_objects: List[NengoGraphObject] = network.all_ensembles + network.all_nodes
    adj: Dict[NengoGraphObject, List[nengo.Connection]] = {obj: [] for obj in all_objects}
    in_degree: Dict[NengoGraphObject, int] = {obj: 0 for obj in all_objects}

    # Build adjacency listing and track entry degrees
    for conn in network.all_connections:
        adj[conn.pre].append(conn)
        in_degree[conn.post] += 1

    # Queue root independent components (nodes/ensembles with 0 incoming dependencies).
    # deque + popleft() keeps this an O(1)-per-pop FIFO queue; list + pop(0) would be O(n)
    # per pop (shifts every remaining element), making the whole sort O(n^2).
    queue: Deque[NengoGraphObject] = deque(obj for obj, deg in in_degree.items() if deg == 0)
    execution_order: List[NengoGraphObject] = []

    while queue:
        curr = queue.popleft()
        execution_order.append(curr)

        for conn in adj[curr]:
            neighbor = conn.post
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If sorted component count doesn't match network total, a loop exists
    if len(execution_order) != len(all_objects):
        problematic_nodes: List[str] = [str(obj.label) for obj, deg in in_degree.items() if deg > 0]
        raise ValueError(
            f"\n[FATAL ERROR] Recurrent Loop Detected in Nengo Network!\n"
            f"TensorFlow Lite Micro cannot compile recurrent memory loops.\n"
            f"The following components are locked in a cycle: {problematic_nodes}\n"
        )

    return execution_order


# =====================================================================
# STAGE 2: Keras Structural Model Builder
# =====================================================================

_T = TypeVar("_T")


def _single_or_list(items: List[_T]) -> Union[_T, List[_T]]:
    """
    Keras's Model(inputs=..., outputs=...) treats a length-1 list differently from a
    bare tensor, so both call sites need to unwrap down to the single item when there's
    only one - this names that check once instead of repeating it per call site.
    """
    return items[0] if len(items) == 1 else items


def build_keras_model_from_nengo(
        sim: nengo.Simulator,
        network: nengo.Network,
        start_nodes: Union[nengo.Node, List[nengo.Node]],
        output_nodes: Union[NengoGraphObject, List[NengoGraphObject]],
        sorted_nodes: List[NengoGraphObject]
) -> tf.keras.Model:
    """
    Parses a validated Nengo DAG and compiles it sequentially into an executable
    TensorFlow Functional Keras Model.
    """
    start_list = start_nodes if isinstance(start_nodes, list) else [start_nodes]
    output_list = output_nodes if isinstance(output_nodes, list) else [output_nodes]

    tensor_map: Dict[NengoGraphObject, tf.Tensor] = {}
    input_tensors: List[tf.Tensor] = []

    # 1. Instantiate Network Entry Ports
    for node in start_list:
        shape = (int(node.size_out),) if hasattr(node, 'size_out') else (1,)
        clean_name = (node.label or f"Input_{id(node)}").replace(" ", "_")
        inp = tf.keras.Input(shape=shape, name=f"Input_{clean_name}")
        input_tensors.append(inp)
        tensor_map[node] = inp

    print(f"[Model Builder] Instantiated {len(input_tensors)} network inputs.")

    # 2. Route Signals Sequentially via Topologically Sorted Elements
    for obj in sorted_nodes:
        if obj in start_list:
            continue

        incoming_conns = [c for c in network.all_connections if c.post == obj]
        if not incoming_conns:
            continue

        branch_outputs: List[tf.Tensor] = []
        for conn in incoming_conns:
            src_tensor = tensor_map.get(conn.pre)
            if src_tensor is None:
                # Should be unreachable given a valid topological order - every predecessor
                # is visited before its successors, so it should already be in tensor_map.
                # Failing loudly here beats silently dropping this branch (and potentially
                # cascading into dropping obj's own successors too) if that guarantee is
                # ever violated by a bug elsewhere.
                raise RuntimeError(
                    f"Internal error: no tensor found for '{str(conn.pre.label)}', "
                    f"predecessor of '{str(obj.label)}', despite processing in "
                    f"topological order. This indicates a bug in the graph walk."
                )

            # If the connection defines custom hardware compilation hooks, apply them.
            # to_keras() returns layers that are already built and weighted.
            if hasattr(conn, 'to_keras'):
                x = src_tensor
                for layer in conn.to_keras(sim):
                    x = layer(x)
                branch_outputs.append(x)
            else:
                branch_outputs.append(src_tensor)

        # Manage structural path convergence (ResNet-style parallel tracking sum)
        if len(branch_outputs) > 1:
            total_input = tf.keras.layers.Add(name=f"ResNet_Sum_{str(obj.label).replace(' ', '_')}")(branch_outputs)
        else:
            total_input = branch_outputs[0]

        # Execute structural translation logic on destination objects.
        # to_keras() returns layers that are already built and weighted.
        if hasattr(obj, 'to_keras'):
            x = total_input
            for layer in obj.to_keras(sim):
                x = layer(x)
            tensor_map[obj] = x
        else:
            tensor_map[obj] = total_input

    # 3. Standardize Final Output Terminals for Clean Netron Visualization Diagrams
    final_outputs = []
    for node in output_list:
        if node in tensor_map:
            clean_name = (node.label or f"Output_{id(node)}").replace(" ", "_")
            # Clear ambiguous trailing names by nesting the final tensor in an identity mapping
            renamed_terminal = tf.keras.layers.Activation('linear', name=clean_name)(tensor_map[node])
            final_outputs.append(renamed_terminal)

    return tf.keras.Model(
        inputs=_single_or_list(input_tensors),
        outputs=_single_or_list(final_outputs)
    )


# =====================================================================
# STAGE 3: TFLITE COMPILATION
# =====================================================================

def compile_saved_model_to_tflite(saved_model_dir: str, tflite_path: str) -> None:
    """
    Invokes the production TFLite compilation subsystem to write out your finalized flatbuffer model binary.
    """
    print("[TFLite Compiler] Compiling saved model to flatbuffer...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    converter.allow_custom_ops = True
    converter.optimizations = []  # Maintain flat structural topologies for hardware register mapping

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path) or ".", exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


# =====================================================================
# PIPELINE ORCHESTRATOR
# =====================================================================

def convert_and_inject_complex_dag(
        sim: nengo.Simulator,
        network: nengo.Network,
        start_nodes: Union[nengo.Node, List[nengo.Node]],
        output_nodes: Union[NengoGraphObject, List[NengoGraphObject]],
        tflite_path: str = "snn.tflite"
) -> None:
    """
    Executes the comprehensive pipeline to transform your architectural Nengo model
    into a custom-op TFLite deployment bundle. The custom hardware ops (and the solved
    constants they carry) are emitted directly by the Keras layers themselves - see
    layers/custom_ops.py - so there is no post-save patching step.
    """
    # Step 1: Model conversion
    sorted_nodes = topological_sort_and_detect_loops(network)
    keras_model = build_keras_model_from_nengo(sim, network, start_nodes, output_nodes, sorted_nodes)
    print("[Pipeline] Keras structural translation complete.")

    # Use secure sandbox memory allocation context for disk-based transformation steps
    temp_dir = tempfile.mkdtemp()
    try:
        # Step 2: Serialize the model to storage disk
        keras_model.save(temp_dir)

        # Step 3: Final flatbuffer generation
        compile_saved_model_to_tflite(temp_dir, tflite_path)
        print(f"[Pipeline] Success! Final compiled binary delivered to -> {tflite_path}")

    finally:
        # Step 4: Clear out temporary scratchpad workspace assets from filesystem
        shutil.rmtree(temp_dir)
