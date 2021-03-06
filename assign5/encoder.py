from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers, layers
from tensorflow.keras.layers import (
    MultiHeadAttention,
    LayerNormalization,
    Dense,
    Conv1D,
    Embedding,
    Dropout,
    Flatten,
    BatchNormalization,
    Activation,
    Add,
    Input,
    GlobalAveragePooling1D,
)
from tensorflow import shape, range
import numpy as np

# https://keras.io/examples/nlp/text_classification_with_transformer/
class Encoder(layers.Layer):
    def __init__(self, num_heads, embedded_dims, feed_forward_dims: list):
        super(Encoder, self).__init__()

        eps = 1e-9
        dropout_rate = 0.3

        self.Attention = MultiHeadAttention(num_heads = num_heads, key_dim = embedded_dims)
        self.Norm1 = LayerNormalization(epsilon = eps)
        self.Norm2 = LayerNormalization(epsilon = eps)
        
        self.FC = Sequential(
                    [Dropout(dropout_rate),
                    Dense(
                        units = feed_forward_dims[0],
                        activation = 'elu',
                        kernel_initializer = 'he_normal',
                        kernel_regularizer = regularizers.l2(0.00001)
                    ), 
                    Dropout(dropout_rate * 2),
                    Dense(
                        units = feed_forward_dims[1],
                        activation = 'elu',
                        kernel_initializer = 'he_normal',
                        kernel_regularizer = regularizers.l2(0.00001)
                    )]
        )

    def call(self, input):
        x = self.Attention(input, input)
        x = self.Norm1(input + x)
        input = x
        x = self.FC(input)
        x = self.Norm2(input + x)
        
        return x

class PositionalEncoder(layers.Layer):
    def __init__(self, vocab_size, max_len, embedded_dims):
        super(PositionalEncoder, self).__init__()
        self.word_embedding = Embedding(input_dim = vocab_size, output_dim = embedded_dims)
        self.pos_embedding = Embedding(input_dim = max_len, output_dim = embedded_dims)

    def call(self, x):  
        max_len = shape(x)[-1]
        pos = range(start = 0, limit = max_len, delta = 1)
        pos = self.pos_embedding(pos)
        x = self.word_embedding(x)
        x += pos

        return x
