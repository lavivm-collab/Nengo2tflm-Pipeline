import tensorflow as tf
import nengo
from typing import List, Union, Dict, Any


# =====================================================================
# 1. GRAPH VALIDATION ENGINE (Cycle Detection)
# =====================================================================
def topological_sort_and_detect_loops(network: nengo.Network) -> List[Union[nengo.Ensemble, nengo.Node]]:
    """
    Performs a topological sort on all components within a Nengo network using
    Kahn's Algorithm, while defending against un-routable recurrent cycles.

    Parameters:
    -----------
    network : nengo.Network
        The complete Nengo network instance containing the blocks and connections.

    Returns:
    --------
    List[Union[nengo.Ensemble, nengo.Node]]
        A linearly ordered sequence of Nengo components ready for execution processing.

    Raises:
    -------
    ValueError
        If a feedback/recurrent loop is detected, preventing linear hardware scheduling.
    """
    all_objects: List[Union[nengo.Ensemble, nengo.Node]] = network.all_ensembles + network.all_nodes
    adj: Dict[Union[nengo.Ensemble, nengo.Node], List[nengo.Connection]] = {obj: [] for obj in all_objects}
    in_degree: Dict[Union[nengo.Ensemble, nengo.Node], int] = {obj: 0 for obj in all_objects}

    # Map connections and calculate incoming dependencies
    for conn in network.all_connections:
        adj[conn.pre].append(conn)
        in_degree[conn.post] += 1

    # Queue up source nodes (components with 0 incoming dependencies)
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

    # --- LOOP DEFENSE CHECK ---
    if len(execution_order) != len(all_objects):
        # Identify the exact nodes stuck in the cycle
        problematic_nodes: List[str] = [str(obj.label) for obj, deg in in_degree.items() if deg > 0]

        raise ValueError(
            f"\n[FATAL ERROR] Recurrent Loop Detected in Nengo Network!\n"
            f"TensorFlow Lite Micro cannot compile recurrent memory loops.\n"
            f"The following components are locked in a cycle: {problematic_nodes}\n"
            f"Please break the feedback loop before deploying to hardware."
        )

    return execution_order


# =====================================================================
# 2. MULTI-IO / RESNET FUNCTIONAL COMPILER
# =====================================================================
def convert_complex_dag(
        sim: nengo.Simulator,
        network: nengo.Network,
        start_nodes: Union[nengo.Node, List[nengo.Node]],
        output_nodes: Union[nengo.Node, nengo.Ensemble, List[Union[nengo.Node, nengo.Ensemble]]],
        tflite_path: str = "snn.tflite"
) -> None:
    """
    Compiles complex Nengo Directed Acyclic Graphs (DAGs) into a deployment-ready
    TensorFlow Lite binary file, natively preserving multi-path signal additions (ResNet paths)
    and multi-input/multi-output configurations.

    Parameters:
    -----------
    sim : nengo.Simulator
        The active Nengo simulator instance holding optimized live network weights.
    network : nengo.Network
        The architectural layout definition containing the SNN components.
    start_nodes : Union[nengo.Node, List[nengo.Node]]
        The entry-point Nengo Node (or list of Nodes) where input signals enter the network.
    output_nodes : Union[nengo.Node, nengo.Ensemble, List[Union[nengo.Node, nengo.Ensemble]]]
        The exit-point component (or list of components) marking the network's processed results.
    tflite_path : str, optional
        The destination storage path for the generated binary flatbuffer model.
        Defaults to "snn.tflite".

    Returns:
    --------
    None
        Writes the final compilation product directly to disk at the designated `tflite_path`.
    """
    # Force inputs and outputs into lists to standardize multi-IO iterations safely
    start_list: List[nengo.Node] = start_nodes if isinstance(start_nodes, list) else [start_nodes]
    output_list: List[Union[nengo.Node, nengo.Ensemble]] = output_nodes if isinstance(output_nodes, list) else [
        output_nodes]

    # Validate graph structure and guard against cycles
    sorted_nodes: List[Union[nengo.Ensemble, nengo.Node]] = topological_sort_and_detect_loops(network)

    # Tensor map serves as our virtual routing breadboard
    tensor_map: Dict[Union[nengo.Ensemble, nengo.Node], tf.Tensor] = {}
    input_tensors: List[tf.Tensor] = []

    # Step 1: Initialize all network entry points (Multiple Inputs)
    for node in start_list:
        shape = (int(node.size_out),) if hasattr(node, 'size_out') else (1,)
        inp = tf.keras.Input(shape=shape, name=f"Input_{node.label}")
        input_tensors.append(inp)
        tensor_map[node] = inp

    print(f"[Compiler] Instantiated {len(input_tensors)} parallel network inputs.")

    # Step 2: Route through the topologically sorted DAG
    for obj in sorted_nodes:
        if obj in start_list:
            continue

        # Collect every incoming branch hitting this component
        incoming_conns: List[nengo.Connection] = [c for c in network.all_connections if c.post == obj]
        if not incoming_conns:
            continue

        branch_outputs: List[tf.Tensor] = []
        for conn in incoming_conns:
            src_tensor = tensor_map.get(conn.pre)
            if src_tensor is None:
                continue  # Path originates from an unmapped or skipped sub-graph region

            # Compile connection modifications (Decoders / Weights / Synapses)
            if hasattr(conn, 'to_keras'):
                layers, weights = conn.to_keras(sim)
                x = src_tensor
                for layer in layers:
                    x = layer(x)
                layers[0].set_weights(weights)
                branch_outputs.append(x)
            else:
                branch_outputs.append(src_tensor)  # Clean wire / skip connection pass-through

        if not branch_outputs:
            continue

        # --- RESNET MERGE / ADDITION ---
        # If multiple branches (like a processing path AND a skip connection) converge, sum them
        if len(branch_outputs) > 1:
            total_input = tf.keras.layers.Add(name=f"ResNet_Sum_{obj.label}")(branch_outputs)
        else:
            total_input = branch_outputs[0]

        # Pass the consolidated signal through the current node's internal hardware operations
        if hasattr(obj, 'to_keras'):
            layers, weights = obj.to_keras(sim)
            x = total_input
            for layer in layers:
                x = layer(x)
            layers[0].set_weights(weights)
            tensor_map[obj] = x
        else:
            tensor_map[obj] = total_input

    # Step 3: Bundle target terminations (Multiple Outputs)
    final_outputs: List[tf.Tensor] = [tensor_map[node] for node in output_list if node in tensor_map]

    # Build Functional Keras model mapping all inputs directly to all outputs
    keras_model = tf.keras.Model(
        inputs=input_tensors if len(input_tensors) > 1 else input_tensors[0],
        outputs=final_outputs if len(final_outputs) > 1 else final_outputs[0]
    )

    print("[Compiler] Structural verification complete. Exporting hardware flatbuffer...")

    # Freeze to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    import numpy as np
    def representative_data_gen():
        for _ in range(200):
            # Yield dummy data matching your input shape and type
            # Replace (1, 10) with your actual input shape
            yield [np.random.uniform(-1, 1, size=(1, 1)).astype(np.float32)]

    converter.representative_dataset = representative_data_gen


    converter.allow_custom_ops = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[Compiler] Success! Compiled file saved to: {tflite_path}")