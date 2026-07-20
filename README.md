# Energy-Efficient Deployment of Spiking Neural Networks on RISC-V

Master's thesis project by Matan Laviv.

The goal of this project is a low-overhead, hardware-software framework for deploying Spiking
Neural Networks (SNNs) on resource-constrained RISC-V edge devices, without the memory or power
overhead of a traditional OS. This repository holds **Phase 1**: a compiler pipeline that
converts [Nengo](https://www.nengo.ai/) SNN models into a bare-metal-deployable artifact — a
TensorFlow Lite Micro (`.tflite`) flatbuffer plus a generated C++ configuration header —
targeting custom hardware LIF/synapse microkernels.

## How it works

Nengo builds SNNs as a graph of `Ensemble`s (populations of spiking neurons) and `Node`s
(inputs/outputs), linked by `Connection`s. The converter ([`src/converter.py`](src/converter.py))
turns that graph into a deployment bundle in five stages:

1. **Topological sort & loop detection** — Kahn's algorithm orders the Nengo DAG. Since TFLM
   cannot route recurrent memory loops as native graph edges, any cycle in the network raises a
   fatal error rather than silently producing a broken model.
2. **Keras model construction** — the sorted graph is rebuilt as a `tf.keras` functional model.
   Nengo objects that expose a `to_keras(sim)` hook (see below) contribute their own Keras layers
   and solved weights (encoders/decoders/biases); everything else passes its tensor through.
3. **Protobuf op patching** — the Keras model is saved as a `SavedModel`, then its protobuf graph
   is walked and placeholder ops are swapped for custom hardware op names (`Sin` →
   `LIFSpikeLayer`, `Cos` → `HardwareSynapseLayer`). Placeholder math ops are used during
   modeling because TFLite's converter would otherwise constant-fold or reject unrecognized ops;
   swapping happens after Keras has done its shape/weight bookkeeping.
4. **TFLite compilation** — the patched `SavedModel` is compiled to a `.tflite` flatbuffer with
   `allow_custom_ops=True` and no graph optimizations, so the hardware op nodes and structure
   survive intact for the target runtime to load.
5. **Hardware header generation** — per-ensemble (`tau_rc`, `tau_ref`, threshold, neuron/dim
   counts) and per-synapse (`tau`) parameters are extracted from the Nengo model and rendered into
   a static `hardware_config.h` via [`src/hardware_config.template`](src/hardware_config.template),
   for the embedded runtime to consume at compile time.

### Custom hardware-mapped Nengo objects

- [`HardwareLIFEnsemble`](src/layers/spike_lif.py) (subclasses `nengo.Ensemble`) — emits a
  `Dense` (encoders + gain, as weights/bias) followed by an `LIFSpikeLayer`, whose `call()` is a
  `sin()` placeholder standing in for the eventual hardware LIF spike/leak/threshold kernel.
- [`HardwareConnection`](src/layers/synapse.py) (subclasses `nengo.Connection`) — emits a `Dense`
  decoder layer, optionally followed by a `SynapseFilterLayer` (`cos()` placeholder for the
  hardware exponential synaptic filter) when the connection has a synapse with a `tau`.

### Example

[`src/two_neurons.py`](src/two_neurons.py) builds a small `Input Node -> HardwareLIFEnsemble(2) ->
HardwareConnection -> Output Node` network, runs a reference Nengo simulation (used to solve
weights and to plot ground-truth spikes/voltage/decoded output), and then calls
`convert_and_inject_complex_dag(...)` to produce `src/dest/two_neurons.tflite` and
`src/dest/hardware_config.h`.

```bash
cd src
python two_neurons.py
```

## Project layout

```
src/
  converter.py                Nengo -> Keras -> TFLite conversion pipeline
  hardware_config.template    Template for the generated C++ hardware config header
  two_neurons.py               Example / integration script
  layers/
    spike_lif.py               HardwareLIFEnsemble + LIFSpikeLayer (Sin placeholder)
    synapse.py                  HardwareConnection + SynapseFilterLayer (Cos placeholder)
  dest/                        Generated output (.tflite + hardware_config.h), gitignored
```

## Requirements

- Python 3.12
- `nengo`
- `tensorflow`
- `matplotlib`, `numpy` (for the example script's plots)

No `requirements.txt` exists yet — install the packages above into your environment.

## Status & roadmap

- ✅ **Phase 1 — Compiler pipeline** (this repo): Nengo DAG → Keras → patched `.tflite` +
  generated hardware config header.
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
