import tensorflow as tf
from tensorflow import keras
import numpy as np

#FFN
def point_wise_feed_forward_network(d_model, dff):
    return keras.Sequential([
      keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


#Position encoding
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

#EncoderLayer
#TRXL-I
class EncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(EncoderLayer, self).__init__()

        self.mha = keras.layers.MultiHeadAttention(num_heads, d_model//num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        attn_input = self.layernorm1(x)
        attn_output = self.mha(attn_input,attn_input,attn_input)  # (batch_size, input_seq_len, d_model)
        out1 = x + attn_output  # (batch_size, input_seq_len, d_model)

        out1_norm = self.layernorm2(out1)
        ffn_output = self.ffn(out1_norm)  # (batch_size, input_seq_len, d_model)
        out2 = out1 + ffn_output  # (batch_size, input_seq_len, d_model)

        return out2

#Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, maximum_position_encoding):
        super(Encoder, self).__init__()
        
        self.prenet = keras.layers.Dense(d_model)
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff)
                           for _ in range(num_layers)]

    def call(self, inputs):
        x = inputs
        seq_len = tf.shape(x)[1]
        # linear embeding
        x = self.prenet(x)
        # position encoding.
        x += self.pos_encoding[:, :seq_len, :]

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # (batch_size, input_seq_len, d_model)

#Transformer Model
class TransformerModel(tf.keras.Model):
    def __init__(self, n_action, num_layers, d_model, num_heads, dff, maximum_position_encoding, dmlp):
        super().__init__(self)
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, maximum_position_encoding)
        self.dense1 = keras.layers.Dense(dmlp, activation='gelu')
        self.dense2 = keras.layers.Dense(n_action)

    def call(self, inputs):
        x = inputs
        x = self.encoder(x)
        x = tf.math.reduce_mean(x, axis = 1)
        x = self.dense1(x)
        out = self.dense2(x)

        return out

#Warmup Learning rate
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)