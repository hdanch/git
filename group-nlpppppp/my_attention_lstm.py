from keras.layers import merge, Bidirectional, Dropout
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
from keras.layers import Masking
from keras.preprocessing import sequence

# if True, the attention vector is shared across the input_dimensions where the attention is applied.
SINGLE_ATTENTION_VECTOR = True
APPLY_ATTENTION_BEFORE_LSTM = False

embedding_dict = dict()
with open('./tweet.model.vec', 'r') as f:
    for line in f.readlines():
        line = line.strip().split(' ')
        if len(line) != 101:
            print('-----------error line--------' + line)
        embedding_dict[line[0]] = map(lambda x:float(x), line[1:])
print(len(embedding_dict))

from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))

def attention_3d_block(inputs, max_words):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, max_words))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(max_words, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def get_data():
    trainX = []
    trainY = []
    max_words = 0
    with open('./tweet_label.txt', 'r') as f:
        for line in f.readlines():
            line = line.strip().split('\t')
            if len(line) != 2:
                print('-------------error line----------' + line)
            text = line[0].strip().split(' ')
            label = int(line[1].replace('__label__', ''))
            words_vec = map(lambda x:embedding_dict.get(x, [0.0]*100), text)
            if len(words_vec) > max_words:
                max_words = len(words_vec)
                continue
            trainX.append(words_vec)
            trainY.append([label])
    return np.array(trainX), np.array(trainY), max_words

def model_attention_applied_after_lstm(max_words):
    inputs = Input(shape=(max_words, 100,))
    #masking = Masking(mask_value=0.0)(inputs)
    lstm_units = 64
    lstm_out = Bidirectional(LSTM(lstm_units, dropout=0.5, return_sequences=True, recurrent_dropout=0.5))(inputs)
    #lstm_out = LSTM(lstm_units, return_sequences=True)(inputs)
    attention_mul = attention_3d_block(lstm_out, max_words)
    attention_mul = Flatten()(attention_mul)
    attention_mul = Dropout(0.5)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model


if __name__ == '__main__':

    trainX, trainY, max_words = get_data()
    trainX = sequence.pad_sequences(trainX, maxlen=max_words, dtype='float64', padding='post')
    print(trainX.shape, trainY.shape)
    if APPLY_ATTENTION_BEFORE_LSTM:
        m = model_attention_applied_before_lstm()
    else:
        m = model_attention_applied_after_lstm(max_words)

    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1])
    #m.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=[f1])
    print(m.summary())
    m.fit(trainX, trainY, epochs=150, batch_size=32, validation_split=0.1)#, validation_data=[testX, testY])

