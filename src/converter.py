import nengo
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
import os
import tempfile
import shutil
from typing import List, Union, Dict


# =====================================================================
# 1. GRAPH VALIDATION ENGINE (Cycle Detection)
# =====================================================================
def topological_sort_and_detect_loops(network: nengo.Network) -> List[Union[nengo.Ensemble, nengo.Node]]:
    """
    Performs a topological sort on all components within a Nengo network using
    Kahn's Algorithm, defending against un-routable recurrent cycles.
    """
    all_objects: List[Union[nengo.Ensemble, nengo.Node]] = network.all_ensembles + network.all_nodes
    adj: Dict[Union[nengo.Ensemble, nengo.Node], List[nengo.Connection]] = {obj: [] for obj in all_objects}
    in_degree: Dict[Union[nengo.Ensemble, nengo.Node], int] = {obj: 0 for obj in all_objects}

    for conn in network.all_connections:
        adj[conn.pre].append(conn)
        in_degree[conn.post] += 1

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

    if len(execution_order) != len(all_objects):
        problematic_nodes: List[str] = [str(obj.label) for obj, deg in in_degree.items() if deg > 0]
        raise ValueError(
            f"\n[FATAL ERROR] Recurrent Loop Detected in Nengo Network!\n"
            f"TensorFlow Lite Micro cannot compile recurrent memory loops.\n"
            f"The following components are locked in a cycle: {problematic_nodes}\n"
        )

    return execution_order


# =====================================================================
# 2. UNIFIED FUNCTIONAL COMPILER & INJECTOR
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
    Compiles complex Nengo Directed Acyclic Graphs (DAGs) into a Functional Keras model,
    surgically injects custom C++ operator definitions into the GraphDef (including the 
    TF2 Function Library), and saves a deployment-ready TFLite binary file.
    """
    # --- PHASE 1: NENGO TO KERAS TRANSLATION ---
    start_list: List[nengo.Node] = start_nodes if isinstance(start_nodes, list) else [start_nodes]
    output_list: List[Union[nengo.Node, nengo.Ensemble]] = output_nodes if isinstance(output_nodes, list) else [
        output_nodes]

    sorted_nodes = topological_sort_and_detect_loops(network)
    tensor_map: Dict[Union[nengo.Ensemble, nengo.Node], tf.Tensor] = {}
    input_tensors: List[tf.Tensor] = []

    for node in start_list:
        shape = (int(node.size_out),) if hasattr(node, 'size_out') else (1,)
        inp = tf.keras.Input(shape=shape, name=f"Input_{node.label}")
        input_tensors.append(inp)
        tensor_map[node] = inp

    print(f"[Compiler] Instantiated {len(input_tensors)} parallel network inputs.")

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

        if len(branch_outputs) > 1:
            total_input = tf.keras.layers.Add(name=f"ResNet_Sum_{obj.label}")(branch_outputs)
        else:
            total_input = branch_outputs[0]

        if hasattr(obj, 'to_keras'):
            layers, weights = obj.to_keras(sim)
            x = total_input
            for layer in layers:
                x = layer(x)
            layers[0].set_weights(weights)
            tensor_map[obj] = x
        else:
            tensor_map[obj] = total_input

    final_outputs = [tensor_map[node] for node in output_list if node in tensor_map]

    keras_model = tf.keras.Model(
        inputs=input_tensors if len(input_tensors) > 1 else input_tensors[0],
        outputs=final_outputs if len(final_outputs) > 1 else final_outputs[0]
    )

    print("[Compiler] Structural translation complete. Initiating deep GraphDef injection...")

    # --- PHASE 3 PRE-REGISTRATION: PREVENT LOADER PANIC ---
    # CRITICAL FIX: The input/output names MUST be 'x' and 'y' to prevent KeyError port mismatches
    custom_opdef = f"""name: '{custom_op_name}'
input_arg: {{ name: 'x' type: DT_FLOAT }}
output_arg: {{ name: 'y' type: DT_FLOAT }}"""

    try:
        from tensorflow.lite.python.convert import register_custom_opdefs
        register_custom_opdefs([custom_opdef])
        print(f"[Compiler] -> Safely pre-registered custom op signature for '{custom_op_name}'")
    except Exception as e:
        print(f"[Compiler] -> Warning during OpDef registration: {e}")

    # --- PHASE 2: DEEP GRAPHDEF SURGICAL INJECTION ---
    temp_dir = tempfile.mkdtemp()
    try:
        keras_model.save(temp_dir)
        saved_model_path = os.path.join(temp_dir, "saved_model.pb")

        sm = saved_model_pb2.SavedModel()
        with tf.io.gfile.GFile(saved_model_path, "rb") as f:
            sm.ParseFromString(f.read())

        patch_count = 0
        for meta_graph in sm.meta_graphs:

            # 1. Scan the main graph blueprint
            for node in meta_graph.graph_def.node:
                if target_namespace.lower() in node.name.lower() and node.op == placeholder_op:
                    print(f"[Injector] -> Patching target node (Main Graph): {node.name}")
                    node.op = custom_op_name
                    patch_count += 1

            # 2. Scan the TF2 Function Library (Where the KeyError was occurring)
            for func in meta_graph.graph_def.library.function:
                for node in func.node_def:
                    if target_namespace.lower() in node.name.lower() and node.op == placeholder_op:
                        print(f"[Injector] -> Patching target node (Function Library): {node.name}")
                        node.op = custom_op_name
                        patch_count += 1

        print(
            f"[Injector] GraphDef patch complete. Rewrote {patch_count} '{placeholder_op}' node(s) to '{custom_op_name}'.")

        with tf.io.gfile.GFile(saved_model_path, "wb") as f:
            f.write(sm.SerializeToString())

        # --- PHASE 3: TFLITE COMPILATION ---
        print("[Compiler] Compiling patched blueprint to TFLite flatbuffer...")
        converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)

        converter.allow_custom_ops = True
        converter.optimizations = []

        tflite_model = converter.convert()

        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print(f"[Compiler] Success! Hardware-ready file saved to: {tflite_path}")

    finally:
        shutil.rmtree(temp_dir)