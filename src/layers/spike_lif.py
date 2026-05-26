import nengo
import numpy as np
import tensorflow as tf


# 1. We still need the Keras "Stub" for TFLite to recognize the op
@tf.keras.utils.register_keras_serializable()
class LIFSpikeLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        return tf.math.sin(inputs)


# 2. The Custom Nengo Component
class HardwareLIFEnsemble(nengo.Ensemble):
    def to_keras(self, sim):
        gains, encoders, biases = sim.data[self].gain, sim.data[self].encoders, sim.data[self].bias
        W = (encoders * gains[:, np.newaxis]).T

        encoder_dense = tf.keras.layers.Dense(self.n_neurons, name=f'{self.label}_Encoders')
        hardware_lif = LIFSpikeLayer(name=f'lif_spike_hardware_node')
        return [encoder_dense, hardware_lif], [W, biases]