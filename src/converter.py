import os
import tempfile
import shutil
from typing import List, Union, Dict, Any
import nengo
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2


# =====================================================================
# STAGE 1: GRAPH TOPOLOGY & LOOP VALIDATION
# =====================================================================

def topological_sort_and_detect_loops(network: nengo.Network) -> List[Union[nengo.Ensemble, nengo.Node]]:
    """
    Performs a topological sort on Nengo network components using Kahn's Algorithm.
    Protects the deployment target by aborting if an un-routable recurrent cycle is detected.
    """
    all_objects: List[Union[nengo.Ensemble, nengo.Node]] = network.all_ensembles + network.all_nodes
    adj: Dict[Union[nengo.Ensemble, nengo.Node], List[nengo.Connection]] = {obj: [] for obj in all_objects}
    in_degree: Dict[Union[nengo.Ensemble, nengo.Node], int] = {obj: 0 for obj in all_objects}

    # Build adjacency listing and track entry degrees
    for conn in network.all_connections:
        adj[conn.pre].append(conn)
        in_degree[conn.post] += 1

    # Queue root independent components (nodes/ensembles with 0 incoming dependencies)
    queue: List[Union[nengo.Ensemble, nengo.Node]] = [obj for obj, deg in in_degree.items() if deg == 0]
    execution_order: List[Union[nengo.Ensemble, nengo.Node]] = []

    while queue:
        curr = queue.pop(0)
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

def build_keras_model_from_nengo(
        sim: nengo.Simulator,
        network: nengo.Network,
        start_nodes: Union[nengo.Node, List[nengo.Node]],
        output_nodes: Union[nengo.Node, nengo.Ensemble, List[Union[nengo.Node, nengo.Ensemble]]]
) -> tf.keras.Model:
    """
    Parses a validated Nengo DAG and compiles it sequentially into an executable
    TensorFlow Functional Keras Model.
    """
    start_list = start_nodes if isinstance(start_nodes, list) else [start_nodes]
    output_list = output_nodes if isinstance(output_nodes, list) else [output_nodes]

    sorted_nodes = topological_sort_and_detect_loops(network)
    tensor_map: Dict[Union[nengo.Ensemble, nengo.Node], tf.Tensor] = {}
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
                continue

            # If the connection defines custom hardware compilation hooks, apply them
            if hasattr(conn, 'to_keras'):
                layers, weights = conn.to_keras(sim)
                x = src_tensor
                for layer in layers:
                    x = layer(x)
                layers[0].set_weights(weights)
                branch_outputs.append(x)
            else:
                branch_outputs.append(src_tensor)

        if not branch_outputs:
            continue

        # Manage structural path convergence (ResNet-style parallel tracking sum)
        if len(branch_outputs) > 1:
            total_input = tf.keras.layers.Add(name=f"ResNet_Sum_{str(obj.label).replace(' ', '_')}")(branch_outputs)
        else:
            total_input = branch_outputs[0]

        # Execute structural translation logic on destination objects
        if hasattr(obj, 'to_keras'):
            layers, weights = obj.to_keras(sim)
            x = total_input
            for layer in layers:
                x = layer(x)
            layers[0].set_weights(weights)
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
        inputs=input_tensors if len(input_tensors) > 1 else input_tensors[0],
        outputs=final_outputs if len(final_outputs) > 1 else final_outputs[0]
    )


# =====================================================================
# STAGE 3: PROTOBUF OPERATION MANIPULATION ENGINE
# =====================================================================

def register_custom_hardware_op(custom_op_name: str) -> None:
    """
    Registers custom hardware execution primitives into the active TensorFlow process
    environment context to defend against graph loading parsing crashes.
    """
    synapse_opdef = (
        "name: 'HardwareSynapseLayer'\n"
        "input_arg: { name: 'x' type: DT_FLOAT }\n"
        "output_arg: { name: 'y' type: DT_FLOAT }"
    )

    lif_opdef = (
        f"name: '{custom_op_name}'\n"
        f"input_arg: {{ name: 'x' type: DT_FLOAT }}\n"
        f"output_arg: {{ name: 'y' type: DT_FLOAT }}"
    )

    try:
        from tensorflow.lite.python.convert import register_custom_opdefs
        register_custom_opdefs([synapse_opdef, lif_opdef])
        print(f"[Registry] Registered custom signatures for 'HardwareSynapseLayer' and '{custom_op_name}'")
    except Exception as e:
        print(f"[Registry] Non-critical warning during custom OpDef allocation step: {e}")


def patch_saved_model_protobuf(
        saved_model_dir: str,
        target_namespace: str,
        placeholder_op: str,
        custom_op_name: str
) -> int:
    """
    Surgically audits SavedModel serialized binary charts, swapping baseline mathematical
    placeholders (Sin/Cos) with target downstream hardware microkernel assignments.
    """
    saved_model_path = os.path.join(saved_model_dir, "saved_model.pb")
    sm = saved_model_pb2.SavedModel()

    with tf.io.gfile.GFile(saved_model_path, "rb") as f:
        sm.ParseFromString(f.read())

    patch_count = 0

    # Unified internal worker function to update computational nodes
    def patch_node_list(nodes):
        nonlocal patch_count
        for node in nodes:
            # Swap Sine operations matching our target namespace namespace with custom LIF layer ops
            if target_namespace.lower() in node.name.lower() and node.op == placeholder_op:
                node.op = custom_op_name
                patch_count += 1
            # Swap Cosine operations within synapse containers with custom Synapse layer ops
            elif "hardware_synapse" in node.name.lower() and node.op == "Cos":
                node.op = "HardwareSynapseLayer"
                patch_count += 1

    for meta_graph in sm.meta_graphs:
        # Route processing through Main Blueprint Nodes
        patch_node_list(meta_graph.graph_def.node)

        # Route processing through Sub-compiled Function Def Library Blocks
        for func in meta_graph.graph_def.library.function:
            patch_node_list(func.node_def)

    # Save modified structural records back out to disk
    with tf.io.gfile.GFile(saved_model_path, "wb") as f:
        f.write(sm.SerializeToString())

    return patch_count


def compile_saved_model_to_tflite(saved_model_dir: str, tflite_path: str) -> None:
    """
    Invokes the production TFLite compilation subsystem to write out your finalized flatbuffer model binary.
    """
    print("[TFLite Compiler] Compiling patched blueprint to flatbuffer...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    converter.allow_custom_ops = True
    converter.optimizations = []  # Maintain flat structural topologies for hardware register mapping

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)


# =====================================================================
# STAGE 4: EMBEDDED C++ CONFIGURATION HEADER COMPILER
# =====================================================================

def generate_hardware_header_from_template(
        network: nengo.Network,
        output_dir: str,
        template_name: str = "hardware_config.template"
) -> None:
    """
    Parses structural parameters from Nengo objects (constants like tau and physical thresholds),
    populating an edge-compilation C++ static array runtime configuration header file.
    """
    sorted_nodes = topological_sort_and_detect_loops(network)

    # 1. Gather Physical Ensemble Constants
    ensembles_found = []
    for obj in sorted_nodes:
        if obj.__class__.__name__ == "HardwareLIFEnsemble" or hasattr(obj, 'neuron_type'):
            tau_rc = getattr(obj.neuron_type, 'tau_rc', 0.02)
            tau_ref = getattr(obj.neuron_type, 'tau_ref', 0.002)
            v_threshold = 1.0

            ensembles_found.append({
                "label": (obj.label or f"ensemble_{id(obj)}").replace(" ", "_"),
                "neurons": obj.n_neurons,
                "dimensions": obj.dimensions,
                "tau_rc": float(tau_rc),
                "tau_ref": float(tau_ref),
                "v_threshold": float(v_threshold)
            })

    # 2. Gather Physical Synaptic Filter Constants
    synapses_found = []
    for conn in network.all_connections:
        if hasattr(conn, 'synapse') and hasattr(conn.synapse, 'tau'):
            pre_label = (conn.pre.label or f"node_{id(conn.pre)}").replace(" ", "_")
            post_label = (conn.post.label or f"node_{id(conn.post)}").replace(" ", "_")
            tau = float(conn.synapse.tau)

            synapses_found.append({
                "pre_label": pre_label,
                "post_label": post_label,
                "tau": tau
            })

    # 3. Locate and Read Template File
    template_path = os.path.join(os.path.dirname(__file__), template_name)
    if not os.path.exists(template_path):
        template_path = template_name

    if not os.path.exists(template_path):
        print(f"[Template Engine] Warning: Could not locate '{template_name}'. Skipping C++ header export.")
        return

    with open(template_path, "r") as f:
        template_content = f.read()

    # 4. Generate C++ Initialization Rows for Output Struct Arrays
    ens_lines = [
        f'    {{ "{e["label"]}", {e["neurons"]}, {e["dimensions"]}, {e["tau_rc"]:.6f}f, {e["tau_ref"]:.6f}f, {e["v_threshold"]:.1f}f }}'
        for e in ensembles_found
    ]
    ens_entries_str = ",\n".join(ens_lines)

    syn_lines = [
        f'    {{ "{s["pre_label"]}", "{s["post_label"]}", {s["tau"]:.6f}f }}'
        for s in synapses_found
    ]
    syn_entries_str = ",\n".join(syn_lines)

    # 5. Populate Structural Fields into Target Format Output File
    final_output = template_content.replace("{NUM_ENSEMBLES}", str(len(ensembles_found)))
    final_output = final_output.replace("{ENSEMBLE_ENTRIES}", ens_entries_str)
    final_output = final_output.replace("{NUM_SYNAPSES}", str(len(synapses_found)))
    final_output = final_output.replace("{SYNAPSE_ENTRIES}", syn_entries_str)

    h_file_path = os.path.join(output_dir, "hardware_config.h")
    with open(h_file_path, "w") as f:
        f.write(final_output)

    print(f"[Template Engine] Statically compiled hardware properties saved to -> {h_file_path}")


# =====================================================================
# STAGE 5: INTEGRATED PIPELINE PIPELINE ORCHESTRATOR
# =====================================================================

def convert_and_inject_complex_dag(
        sim: nengo.Simulator,
        network: nengo.Network,
        start_nodes: Union[nengo.Node, List[nengo.Node]],
        output_nodes: Union[nengo.Node, nengo.Ensemble, List[Union[nengo.Node, nengo.Ensemble]]],
        target_namespace: str,
        placeholder_op: str = "Sin",
        custom_op_name: str = "LIFSpikeLayer",
        tflite_path: str = "snn.tflite"
) -> None:
    """
    Executes the comprehensive pipeline to transform your architectural Nengo model
    into a custom hardware-accelerated TFLite deployment bundle.
    """
    # Step 1: Model conversion
    keras_model = build_keras_model_from_nengo(sim, network, start_nodes, output_nodes)
    print("[Pipeline] Keras structural translation complete.")

    # Step 2: Custom Op Environment Signatures Registration
    register_custom_hardware_op(custom_op_name)

    # Use secure sandbox memory allocation context for disk-based transformation steps
    temp_dir = tempfile.mkdtemp()
    try:
        # Step 3: Serialize unpatched model configuration structures to storage disk
        keras_model.save(temp_dir)

        # Step 4: Run Protobuf Deep-Patcher tool to map hardware execution ops
        patches = patch_saved_model_protobuf(temp_dir, target_namespace, placeholder_op, custom_op_name)
        print(f"[Pipeline] Deep patch complete. Mutated {patches} nodes to hardware operations.")

        # Step 5: Final flatbuffer generation
        compile_saved_model_to_tflite(temp_dir, tflite_path)
        print(f"[Pipeline] Success! Final compiled binary delivered to -> {tflite_path}")

        # Step 6: Export C++ parameters header file
        output_directory = os.path.dirname(tflite_path) or "."
        generate_hardware_header_from_template(network, output_directory)

    finally:
        # Step 7: Clear out temporary scratchpad workspace assets from filesystem
        shutil.rmtree(temp_dir)