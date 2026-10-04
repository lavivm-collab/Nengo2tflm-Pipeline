"""
The two custom hardware ops as real TensorFlow graph ops.

An OpDef registered via register_custom_opdefs() lands in TensorFlow's global op registry. That is
all Keras tracing, SavedModel export and TFLite conversion need - none of them execute the op, so no
compiled kernel exists or is needed (the real kernels are the TFLM ones on the device).

Declaring the constants as typed `attr` fields is what makes them survive into the .tflite
custom_options FlexBuffer that a TFLM kernel reads in Init().
"""
from tensorflow.lite.python.convert import register_custom_opdefs
from tensorflow.python.framework import op_def_library

LIF_OP_NAME = "LIFSpikeLayer"
SYNAPSE_OP_NAME = "HardwareSynapseLayer"

_OPDEFS = {
    LIF_OP_NAME: (
        f"name: '{LIF_OP_NAME}'\n"
        "input_arg: { name: 'x' type: DT_FLOAT }\n"
        "output_arg: { name: 'y' type: DT_FLOAT }\n"
        "attr: { name: 'tau_rc' type: 'float' default_value: { f: 0.02 } }\n"
        "attr: { name: 'tau_ref' type: 'float' default_value: { f: 0.002 } }\n"
        "attr: { name: 'v_threshold' type: 'float' default_value: { f: 1.0 } }\n"
        "attr: { name: 'dt' type: 'float' default_value: { f: 0.001 } }"
    ),
    SYNAPSE_OP_NAME: (
        f"name: '{SYNAPSE_OP_NAME}'\n"
        "input_arg: { name: 'x' type: DT_FLOAT }\n"
        "output_arg: { name: 'y' type: DT_FLOAT }\n"
        "attr: { name: 'tau' type: 'float' default_value: { f: 0.01 } }\n"
        "attr: { name: 'dt' type: 'float' default_value: { f: 0.001 } }"
    ),
}

_registered = set()


def _emit(op_name, x, **attrs):
    # Registering the same op twice aborts the whole process (a C++ CHECK, not a catchable
    # exception), so each op is registered exactly once, on first use.
    if op_name not in _registered:
        register_custom_opdefs([_OPDEFS[op_name]])
        _registered.add(op_name)
    y = op_def_library.apply_op(op_name, x=x, **attrs)
    y.set_shape(x.shape)
    return y


def lif_spike_op(x, tau_rc, tau_ref, v_threshold, dt):
    return _emit(LIF_OP_NAME, x, tau_rc=tau_rc, tau_ref=tau_ref, v_threshold=v_threshold, dt=dt)


def hardware_synapse_op(x, tau, dt):
    return _emit(SYNAPSE_OP_NAME, x, tau=tau, dt=dt)
