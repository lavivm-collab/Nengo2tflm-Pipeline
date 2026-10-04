# Energy-Efficient Deployment of Spiking Neural Networks on RISC-V

Master's thesis project by Matan Laviv.

The goal of this project is a low-overhead, hardware-software framework for deploying Spiking
Neural Networks (SNNs) on resource-constrained RISC-V edge devices, without the memory or power
overhead of a traditional OS. This repository holds **Phase 1**: a compiler pipeline that
converts [Nengo](https://www.nengo.ai/) SNN models into a bare-metal-deployable artifact — a
self-contained TensorFlow Lite Micro (`.tflite`) flatbuffer, with all neuron/synapse constants
embedded directly in it — targeting custom hardware LIF/synapse microkernels.

## How it works

Nengo builds SNNs as a graph of `Ensemble`s (populations of spiking neurons) and `Node`s
(inputs/outputs), linked by `Connection`s. The converter ([`src/converter.py`](src/converter.py))
turns that graph into a deployment bundle in three stages:

1. **Topological sort & loop detection** — Kahn's algorithm orders the Nengo DAG. Since TFLM
   cannot route recurrent memory loops as native graph edges, any cycle in the network raises a
   fatal error rather than silently producing a broken model.
2. **Keras model construction** — the sorted graph is rebuilt as a `tf.keras` functional model.
   Nengo objects that expose a `to_keras(sim)` hook (see below) contribute their own Keras layers
   and solved weights (encoders/decoders/biases); everything else passes its tensor through.
   The custom layers emit the real hardware ops (`LIFSpikeLayer`, `HardwareSynapseLayer`)
   directly, each carrying its own solved constants (`tau_rc`, `tau_ref`, `v_threshold`, `tau`,
   and the simulation `dt`) as typed op attrs. The ops are declared once in
   [`src/layers/custom_ops.py`](src/layers/custom_ops.py) via `register_custom_opdefs`, which
   puts the OpDef in TensorFlow's global op registry — enough for Keras tracing, `SavedModel`
   export and TFLite conversion, none of which execute the op, so no TensorFlow kernel exists
   (the real kernels are the TFLM ones on the device). Declaring the constants as OpDef `attr`
   fields is what lets them survive into the final flatbuffer's `custom_options` (a FlexBuffer a
   TFLM kernel can read at `Init()`).
3. **TFLite compilation** — the Keras model is saved as a `SavedModel` and compiled to a
   `.tflite` flatbuffer with `allow_custom_ops=True` and no graph optimizations, so the hardware
   op nodes, their structure, and their embedded per-op constants survive intact for the target
   runtime to load.

### Custom hardware-mapped Nengo objects

- [`HardwareLIFEnsemble`](src/layers/spike_lif.py) (subclasses `nengo.Ensemble`) — emits a
  `Dense` (encoders + gain, as weights/bias) followed by an `LIFSpikeLayer`, whose `call()` emits
  the `LIFSpikeLayer` custom op (the hardware LIF spike/leak/threshold kernel on the device).
- [`HardwareConnection`](src/layers/synapse.py) (subclasses `nengo.Connection`) — emits a `Dense`
  decoder layer, optionally followed by a `SynapseFilterLayer` (emits the `HardwareSynapseLayer`
  custom op, the hardware exponential synaptic filter) when the connection has a synapse with a
  `tau`.

### Example

[`src/two_neurons.py`](src/two_neurons.py) builds a small `Input Node -> HardwareLIFEnsemble(2) ->
HardwareConnection -> Output Node` network, runs a reference Nengo simulation (used to solve
weights and to plot ground-truth spikes/voltage/decoded output), and then calls
`convert_and_inject_complex_dag(...)` to produce `src/dest/two_neurons.tflite`.
[`src/two_neurons_2d.py`](src/two_neurons_2d.py) is the same example with a 2-dimensional
ensemble instead of 1D, verifying the pipeline generalizes beyond scalar representations.

```bash
cd src
python two_neurons.py
```

## Project layout

```
src/
  converter.py                Nengo -> Keras -> TFLite conversion pipeline
  two_neurons.py               Example / integration script (1D)
  two_neurons_2d.py            Same example with a 2D ensemble
  layers/
    custom_ops.py              Custom op declarations (OpDefs), one-time registration, op emitters
    spike_lif.py               HardwareLIFEnsemble + LIFSpikeLayer
    synapse.py                  HardwareConnection + SynapseFilterLayer
  dest/                        Generated .tflite output, gitignored
```

## Requirements

- Python 3.12
- `nengo`
- `tensorflow`
- `matplotlib`, `numpy` (for the example script's plots)

No `requirements.txt` exists yet — install the packages above into your environment.

## Status & roadmap

- ✅ **Phase 1 — Compiler pipeline** (this repo): Nengo DAG → Keras → self-describing
  `.tflite` (neuron/synapse constants embedded as custom-op attrs, no separate config file).
- ⏳ **Phase 2 — Bare-metal exploration & microkernel optimization**: implement the actual
  `LIFSpikeLayer`/`HardwareSynapseLayer` TFLM C++ kernels (`Init`/`Prepare`/`Invoke`, static
  tensor-arena state for membrane voltage and refractory counters, no heap allocation), a
  bare-metal runner (`main.cc`) registering the ops via `MicroMutableOpResolver`, and RISC-V
  vector-extension / MMIO-accelerator profiling. Not yet present in this repository.

## Constraints for downstream (embedded) implementation

- No dynamic allocation — all per-neuron/per-synapse state must live in the TFLM tensor arena.
- Strict forward-routed DAG — no native recurrent graph edges; time-series state (voltage,
  refractory counters) is tracked inside operator state, not the graph.
- `Invoke` execution loops should stay structured and explicit so blocks can later be swapped for
  RISC-V vector extension (RVV) or MMIO accelerator commands.
- Per-op configuration (`tau_rc`, `tau_ref`, `v_threshold`, `tau`, `dt`) is not passed as a
  separate file — it's read from each op's `custom_options` FlexBuffer at `Init()` time (e.g. via
  `flexbuffers::GetRoot(buffer, length).AsMap()["tau_rc"].AsFloat()`), same as any standard TFLite
  custom op.
