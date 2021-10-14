from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, SpatialDropout1D, Dropout, LSTM, SimpleRNN, GRU
from tensorflow.keras import regularizers
import numpy as np
import re

from parser import CLI_Parser
from parser import Parser
from resnet1D import ResNet_N
from encoder import Encoder, PositionalEncoder

BATCH_SIZE = 256
EPOCHS = 10
EMBED_DIMS = 32

# takes in pandas dataframe of text data
# https://medium.com/@saitejaponugoti/nlp-natural-language-processing-with-tensorflow-b2751aa8c460
# https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
def text_processing(train, test):
    print("[START]: Processing text input...")
    train_data, train_labels = train[-2::-1]
    test_data, test_labels = test[-2::-1]
    train_labels = to_categorical(train_labels - 1)
    test_labels = to_categorical(test_labels - 1)
    train_data = [re.sub(r'[^a-z\d\s]', '', string.lower()) for string in train_data]
    max_len = len(max(train_data, key=len).split())

    word_tokenizer = Tokenizer(oov_token='<OOV>')
    word_tokenizer.fit_on_texts(train_data)
    train_data = word_tokenizer.texts_to_sequences(train_data)
    train_size = len(word_tokenizer.word_index) + 1 # +1 for unknown words
    train_data = pad_sequences(train_data, padding='post', maxlen=max_len)
    test_data = word_tokenizer.texts_to_sequences(test_data)
    test_data = pad_sequences(test_data, padding='post', maxlen=max_len)
    
    print("[END]: Processing successful.")
    return (train_data, train_labels, train_size, max_len, test_data, test_labels)


def main():
    args = CLI_Parser()()
    parser = Parser(args.train, args.test)
    train, test = parser._getAtrributes()
    train_data, train_labels, train_size, max_len, test_data, test_labels = text_processing(train, test)
    STEPS = 0.8 * train_data.shape[0] // BATCH_SIZE

    """model = ResNet_N(doc_size = train_size,
                     max_len = max_len,
                     layers = [2, 2, 2, 2],
                     classes = 4)"""
    """
    model = Sequential()
    model.add(Embedding(input_dim = train_size,
                  output_dim = 32,
                  input_length = max_len))
    model.add(GRU(128, return_sequences=True))
    model.add(SimpleRNN(64, return_sequences=True))
    model.add(SpatialDropout1D(0.5))
    model.add(Conv1D(filters = 128,
                     kernel_size = 5,
                     strides = 2,
                     activation = 'elu',
                     kernel_initializer = 'he_normal',
                     kernel_regularizer=regularizers.l2(0.001)))
    model.add(SpatialDropout1D(0.5))
    model.add(Conv1D(filters = 128,
                     kernel_size = 5,
                     activation = 'elu',
                     kernel_initializer = 'he_normal',
                     kernel_regularizer=regularizers.l2(0.001)))
    model.add(MaxPooling1D(padding='same'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dropout(0.25))
    model.add(Dense(units = 4,
              activation = 'softmax'))
    """

    input = Input(max_len)
    x = PositionalEncoder(vocab_size = train_size,
                          max_len = max_len, 
                          embedded_dims = EMBED_DIMS)(input)
    x = Encoder(num_heads = 2,
                embedded_dims = EMBED_DIMS,
                feed_forward_dims = [128, EMBED_DIMS])(x)
    x = SpatialDropout1D(0.2)(x)
    x = Conv1D(filters = 32,
               kernel_size = 1,
               activation = 'elu',
               kernel_initializer = 'he_normal', 
               kernel_regularizer = regularizers.l2(0.0001))(x)
    x = SpatialDropout1D(0.3)(x)
    x = Conv1D(filters = 64,
               kernel_size = 2,
               activation = 'elu',
               kernel_initializer = 'he_normal', 
               kernel_regularizer = regularizers.l2(0.0001))(x)
    x = SpatialDropout1D(0.4)(x)
    x = Conv1D(filters = 128,
               kernel_size = 3,
               activation = 'elu',
               kernel_initializer = 'he_normal', 
               kernel_regularizer = regularizers.l2(0.0001))(x)
    x = SpatialDropout1D(0.5)(x)
    x = Conv1D(filters = 256,
               kernel_size = 4,
               activation = 'elu',
               kernel_initializer = 'he_normal', 
               kernel_regularizer = regularizers.l2(0.0001))(x)
    x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    x = Dropout(0.3)(x)
    x = Dense(50, activation='elu')(x)
    x = Dropout(0.1)(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    # compile and fit
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(
			x=train_data,
			y=train_labels,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			steps_per_epoch=STEPS,
			validation_split=0.2
	)

    model.evaluate(x=test_data, y=test_labels)


if __name__ == "__main__":
    main()


    

