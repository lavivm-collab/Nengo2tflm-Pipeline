import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2
import os
import tempfile
import shutil


def export_injected_tflite(
        keras_model: tf.keras.Model,
        tflite_path: str,
        target_namespace: str,
        placeholder_op: str,
        custom_op_name: str
) -> None:
    """
    Saves a Keras model, surgically injects a custom C++ operator name into the
    GraphDef architecture, and compiles it to a TFLite flatbuffer.

    Parameters:
    -----------
    keras_model : tf.keras.Model
        The functional Keras model ready for export.
    tflite_path : str
        The destination path for the final .tflite file.
    target_namespace : str
        The substring in the node name to target (e.g., 'lif_spike').
    placeholder_op : str
        The dummy math operation used in Python (e.g., 'Sin').
    custom_op_name : str
        The true name expected by the C++ microcontroller runtime (e.g., 'LIFSpikeLayer').
    """
    # Create a temporary directory to host the intermediate SavedModel
    temp_dir = tempfile.mkdtemp()

    try:
        # Step 1: Export raw Keras model
        keras_model.save(temp_dir)
        print(f"[Injector] Keras model staged in temporary memory.")

        # Step 2: Load the architectural blueprint
        saved_model_path = os.path.join(temp_dir, "saved_model.pb")
        sm = saved_model_pb2.SavedModel()

        with tf.io.gfile.GFile(saved_model_path, "rb") as f:
            sm.ParseFromString(f.read())

        # Step 3: Execute the Surgical Strike
        patch_count = 0
        for meta_graph in sm.meta_graphs:
            for node in meta_graph.graph_def.node:
                # Check for our specific layer AND the placeholder op
                if target_namespace.lower() in node.name.lower() and node.op == placeholder_op:
                    print(f"[Injector] -> Patching target node: {node.name}")
                    node.op = custom_op_name
                    patch_count += 1

        print(
            f"[Injector] GraphDef patch complete. {patch_count} '{placeholder_op}' node(s) rewritten to '{custom_op_name}'.")

        # Step 4: Overwrite the blueprint with the mutated graph
        with tf.io.gfile.GFile(saved_model_path, "wb") as f:
            f.write(sm.SerializeToString())

        # Step 5: Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(temp_dir)
        converter.allow_custom_ops = True
        converter.optimizations = []  # Ensure no fusions break our custom op

        tflite_model = converter.convert()

        # Step 6: Save the final binary
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)

        print(f"[Injector] Success! Binary flatbuffer saved to: {tflite_path}")

    finally:
        # Cleanup: Always delete the temporary directory, even if an error occurs
        shutil.rmtree(temp_dir)