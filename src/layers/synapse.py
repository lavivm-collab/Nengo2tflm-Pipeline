import nengo
import numpy as np
import tensorflow as tf

# 1. The Keras "Stub" for the Synapse Op
@tf.keras.utils.register_keras_serializable()
class SynapseFilterLayer(tf.keras.layers.Layer):
    def __init__(self, tau=0.01, dt=0.001, **kwargs):
        super(SynapseFilterLayer, self).__init__(**kwargs)
        self.tau, self.dt = tau, dt
    def call(self, inputs): return tf.identity(inputs)
    def get_config(self):
        config = super().get_config()
        config.update({"tau": self.tau, "dt": self.dt})
        return config


# 2. The Custom Nengo Component
class HardwareConnection(nengo.Connection):
    def to_keras(self, sim):
        W = sim.data[self].weights.T
        B = np.zeros(self.post.size_in)
        tau = self.synapse.tau if hasattr(self.synapse, 'tau') else 0.01

        decoder_dense = tf.keras.layers.Dense(self.post.size_in, name=f'Decoders_to_{self.post.label}')
        hardware_synapse = SynapseFilterLayer(tau=tau, dt=0.001, name=f'Hardware_Synapse')
        return [decoder_dense, hardware_synapse], [W, B]