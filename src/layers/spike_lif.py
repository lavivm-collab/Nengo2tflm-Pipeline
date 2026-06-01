import nengo
import numpy as np
import tensorflow as tf

@tf.keras.utils.register_keras_serializable()
class LIFSpikeLayer(tf.keras.layers.Layer):
    def __init__(self, tau_rc=0.02, tau_ref=0.002, v_threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.tau_rc = float(tau_rc)
        self.tau_ref = float(tau_ref)
        self.v_threshold = float(v_threshold)

    def call(self, inputs):
        return tf.math.sin(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tau_rc": self.tau_rc,
            "tau_ref": self.tau_ref,
            "v_threshold": self.v_threshold
        })
        return config

class HardwareLIFEnsemble(nengo.Ensemble):
    def to_keras(self, sim):
        gains, encoders, biases = sim.data[self].gain, sim.data[self].encoders, sim.data[self].bias
        W = (encoders * gains[:, np.newaxis]).T

        tau_rc = getattr(self.neuron_type, 'tau_rc', 0.02)
        tau_ref = getattr(self.neuron_type, 'tau_ref', 0.002)
        v_threshold = 1.0

        clean_label = (self.label or f"ensemble_{id(self)}").replace(" ", "_")

        # Make the layer names distinct using the unique ensemble labels
        encoder_dense = tf.keras.layers.Dense(self.n_neurons, name=f'{clean_label}_Encoders')
        hardware_lif = LIFSpikeLayer(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            v_threshold=v_threshold,
            name=f'{clean_label}_lif_spike_hardware_node'
        )
        return [encoder_dense, hardware_lif], [W, biases]