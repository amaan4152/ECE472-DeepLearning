from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, SpatialDropout1D, Dropout, LSTM, SimpleRNN, GRU
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

from parser import CLI_Parser
from parser import Parser
from encoder import Encoder, PositionalEncoder

BATCH_SIZE = 512
EPOCHS = 10
EMBED_DIMS = 100

# https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
def plot_diagnostics(history):
    # plot loss
    plt.subplot(211)
    plt.title("Cross Entropy Loss")
    plt.plot(history.history["loss"], color="blue", label="train")
    plt.plot(history.history["val_loss"], color="orange", label="validation")
    # plot accuracy
    plt.subplot(212)
    plt.title("Classification Accuracy")
    plt.plot(history.history["accuracy"], color="blue", label="train")
    plt.plot(history.history["val_accuracy"], color="orange", label="validation")
    plt.legend()
    plt.tight_layout()
    # save plot to file
    plt.savefig(
        "/zooper2/amaan.rahman/ECE472-DeepLearning/assign5/diagnostics_plot_00.png"
    )
    plt.close()

# takes in pandas dataframe of text data
# https://medium.com/@saitejaponugoti/nlp-natural-language-processing-with-tensorflow-b2751aa8c460
# https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/
def text_processing(train, test):
    print("[START]: Processing text input...")
    train_data, train_labels = train[-2::-1]
    test_data, test_labels = test[-2::-1]
    train_labels = to_categorical(train_labels - 1)
    test_labels = to_categorical(test_labels - 1)

    word_tokenizer = Tokenizer(oov_token='<OOV>')
    word_tokenizer.fit_on_texts(train_data)
    train_data = word_tokenizer.texts_to_sequences(train_data)
    test_data = word_tokenizer.texts_to_sequences(test_data)

    max_len_train = max([len(x) for x in train_data])
    max_len_test = max([len(x) for x in test_data])
    max_len = max_len_train if max_len_train > max_len_test else max_len_test
    train_size = len(word_tokenizer.word_index) + 1 # +1 for unknown words

    train_data = pad_sequences(train_data, padding='post', maxlen=max_len)
    test_data = pad_sequences(test_data, padding='post', maxlen=max_len)
    
    print("[END]: Processing successful.")
    return (train_data, train_labels, train_size, max_len, test_data, test_labels)


def main():
    args = CLI_Parser()()
    parser = Parser(args.train, args.test)
    train, test = parser._getAtrributes()
    train_data, train_labels, train_size, max_len, test_data, test_labels = text_processing(train, test)
    STEPS = 0.8 * train_data.shape[0] // BATCH_SIZE

    input = Input(max_len)
    x = PositionalEncoder(vocab_size = train_size,
                          max_len = max_len, 
                          embedded_dims = EMBED_DIMS)(input)
    x = SpatialDropout1D(0.8)(x)
    x = Conv1D(filters = 512,
               kernel_size = 2,
               activation = 'elu',
               kernel_initializer = 'he_normal', 
               kernel_regularizer = regularizers.l2(0.001))(x)
    x = SpatialDropout1D(0.5)(x)
    x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1000, activation='elu', kernel_regularizer = regularizers.l2(0.0001), kernel_initializer = 'he_normal')(x)
    x = Dropout(0.3)(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    model.summary()

    # compile and fit
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    history = model.fit(
			x=train_data,
			y=train_labels,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			steps_per_epoch=STEPS,
			validation_split=0.2
	)

    model.evaluate(x=test_data, y=test_labels)
    plot_diagnostics(history)

if __name__ == "__main__":
    main()


    

