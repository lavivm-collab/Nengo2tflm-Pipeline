import os
import numpy as np

import nengo
from nengo.dists import Uniform

from layers.spike_lif import HardwareLIFEnsemble
from layers.synapse import HardwareConnection
from src.converter import convert_and_inject_complex_dag

model = nengo.Network(label="Two Neurons 2D")
with model:
    # dimensions=2 instead of 1: encoders must now be (n_neurons, 2) unit vectors,
    # one axis per neuron here (neuron 0 -> x-axis, neuron 1 -> y-axis).
    neurons = HardwareLIFEnsemble(
        2,
        dimensions=2,
        intercepts=Uniform(-0.5, -0.5),
        max_rates=Uniform(100, 100),
        encoders=[[1, 0], [0, 1]],
        label="Sine_Neurons_2D"
    )

    # Input node must now output 2 values to match neurons.size_in == 2.
    sin = nengo.Node(lambda t: [np.sin(8 * t), np.cos(8 * t)], label="Input_Node")
    nengo.Connection(sin, neurons, synapse=0.01)

    # Output node must now accept 2 values to match neurons.size_out == 2.
    out = nengo.Node(size_in=2, label="Output_Node")
    out_conn = HardwareConnection(neurons, out, synapse=0.01)

    sin_probe = nengo.Probe(sin)
    spikes = nengo.Probe(neurons.neurons)
    voltage = nengo.Probe(neurons.neurons, "voltage")
    filtered = nengo.Probe(neurons, synapse=0.01)

with nengo.Simulator(model) as sim:
    sim.run(1)

print(f"Input node size_out:  {sin.size_out}")
print(f"Ensemble dimensions:  {neurons.dimensions}  (n_neurons={neurons.n_neurons})")
print(f"Output node size_in:  {out.size_in}")
print(f"Decoded output shape sim.data[filtered]: {sim.data[filtered].shape}")
print(f"Spike data shape sim.data[spikes]:       {sim.data[spikes].shape}")

print("\n--- HardwareLIFEnsemble.to_keras(sim) ---")
ens_layers = neurons.to_keras(sim)
W_enc, biases = ens_layers[0].get_weights()
print(f"encoder Dense kernel W shape: {W_enc.shape}  (expected (dimensions={neurons.dimensions}, n_neurons={neurons.n_neurons}))")
print(f"encoder Dense bias shape:     {biases.shape}  (expected (n_neurons={neurons.n_neurons},))")

print("\n--- HardwareConnection.to_keras(sim) ---")
conn_layers = out_conn.to_keras(sim)
W_dec, B_dec = conn_layers[0].get_weights()
print(f"decoder Dense kernel W shape: {W_dec.shape}  (expected (pre.size_out={neurons.size_out}, post.size_in={out.size_in}))")
print(f"decoder Dense bias shape:     {B_dec.shape}  (expected (post.size_in={out.size_in},))")

os.makedirs("dest", exist_ok=True)
convert_and_inject_complex_dag(
    sim=sim,
    network=model,
    start_nodes=sin,
    output_nodes=out,
    target_namespace="lif_spike",
    placeholder_op="Sin",
    custom_op_name="LIFSpikeLayer",
    tflite_path="dest/two_neurons_2d.tflite"
)
