from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np
import re

from parser import CLI_Parser
from parser import Parser

from tensorflow.python.ops.gen_array_ops import Empty

# takes in pandas dataframe of text data
# https://medium.com/@saitejaponugoti/nlp-natural-language-processing-with-tensorflow-b2751aa8c460
# https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
def text_processing(train, test):
    print("[START]: Processing text input...")
    train_data, train_labels = train[-2::-1]
    test_data, test_labels = test[-2::-1]
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    train_data = [re.sub(r'[^a-z\d\s]', '', string.lower()) for string in train_data]
    max_len = len(max(train_data, key=len).split())

    word_tokenizer = Tokenizer(oov_token='<OOV>')
    word_tokenizer.fit_on_texts(train_data)
    train_data = word_tokenizer.texts_to_sequences(train_data)
    train_data = pad_sequences(train_data, padding='post')
    test_data = word_tokenizer.texts_to_sequences(test_data)
    test_data = pad_sequences(test_data, padding='post')

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    train_size = len(word_tokenizer.word_index) + 1 # +1 for unknown words
    
    print("[END]: Processing successful.")
    return (train_data, train_labels, train_size, max_len, test_data, test_labels)



def main():
    args = CLI_Parser()()
    parser = Parser(args.train, args.test)
    train, test = parser._getAtrributes()
    train_data, train_labels, train_size, max_len, test_data, test_labels = text_processing(train, test)

    input = Input(train_data.shape)
    x = input
    x = Embedding(input_dim = train_size,
                  output_dim = 100,
                  input_length = max_len)(x)

    x = Conv1D(filters = 128,
               kernel_size = 4,
               activation = 'relu',
               kernel_initializer = 'he_normal')(x)
    
    x = MaxPooling1D()(x)
    x = Flatten()(x)
    x = Dense(units = 100,
              activation = 'relu')(x)

    x = Dense(units = 4,
              activation = 'relu')(x)
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


if __name__ == "__main__":
    main()


    

