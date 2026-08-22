import nengo
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class SynapseFilterLayer(tf.keras.layers.Layer):
    def __init__(self, tau=0.01, dt=0.001, size_in=1, **kwargs):
        super(SynapseFilterLayer, self).__init__(**kwargs)
        self.tau = float(tau)
        self.dt = float(dt)
        self.size_in = int(size_in)

    def call(self, inputs):
        # Use Cosine as a 1-to-1 placeholder!
        # TFLite won't optimize it away, and it maps perfectly to our 1-input custom op.
        return tf.math.cos(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "tau": self.tau,
            "dt": self.dt,
            "size_in": self.size_in
        })
        return config


class HardwareConnection(nengo.Connection):
    def to_keras(self, sim):
        # 1. Extract or generate weights
        nengo_weights = sim.data[self].weights
        if nengo_weights is None:
            W = np.eye(self.post.size_in, self.pre.size_out, dtype=np.float32)
        else:
            W = nengo_weights.T

        B = np.zeros(self.post.size_in, dtype=np.float32)

        # 2. Format names safely
        pre_label = (self.pre.label or f"node_{id(self.pre)}").replace(" ", "_")
        post_label = (self.post.label or f"node_{id(self.post)}").replace(" ", "_")

        # 3. Always instantiate the decoder Dense layer
        decoder_dense = tf.keras.layers.Dense(
            self.post.size_in,
            name=f'Decoders_{pre_label}_to_{post_label}'
        )
        # Build with the known input shape and inject the solved decoder weights immediately,
        # rather than leaving the layer unbuilt (random weights) for the caller to fix up after
        # the fact - a forgotten set_weights() call would otherwise silently ship a model with
        # meaningless random decoders instead of the solved NEF values, with no error at all.
        # When pre is an ensemble, the tensor this layer actually receives is the neuron
        # activation output (width n_neurons), not the ensemble's decoded dimensionality -
        # decoding happens here, in the connection, not inside the ensemble's own to_keras().
        pre_width = self.pre.n_neurons if hasattr(self.pre, 'n_neurons') else self.pre.size_out
        decoder_dense.build((None, pre_width))
        decoder_dense.set_weights([W, B])

        # Start our layers array with just the dense layer
        layers = [decoder_dense]

        # 4. DYNAMIC FIX: Only instantiate and add the synapse layer if a synapse exists
        if self.synapse is not None and hasattr(self.synapse, 'tau'):
            tau = float(self.synapse.tau)
            hardware_synapse = SynapseFilterLayer(
                tau=tau,
                dt=0.001,
                size_in=self.pre.size_out,
                name=f'Hardware_Synapse_{pre_label}_to_{post_label}'
            )
            layers.append(hardware_synapse)

        # Return the dynamically built layers - already fully weighted, no separate injection step
        return layers