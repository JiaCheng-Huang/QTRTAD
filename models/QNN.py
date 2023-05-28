# -*- coding: utf-8 -*-

from keras import Model
from keras.engine.saving import load_model
from keras.layers import *
from keras.optimizers import RMSprop
from keras_preprocessing.sequence import pad_sequences

from layers.embedding import phase_embedding_layer, amplitude_embedding_layer
from layers.measurement import ComplexMeasurement
from layers.mixture import ComplexMixture
from layers.multiply import ComplexMultiply
from utils import split_texts


class QNN:
    def __init__(self, embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE, CLAUSE_NUM):
        self.embedding_matrix = embedding_matrix
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.VOCAB_SIZE = VOCAB_SIZE
        self.CLAUSE_NUM = CLAUSE_NUM
        self.weight_embedding = Embedding(self.embedding_matrix.shape[0], 1, trainable=True)
        self.model = self.build_model()

    def build_model(self):
        input1 = Input(shape=(self.seq_length,), dtype='int32')
        input2 = Input(shape=(self.seq_length,), dtype='int32')

        weight1 = Activation('softmax')(self.weight_embedding(input1))
        phase_encoded1 = phase_embedding_layer(self.seq_length, self.embedding_matrix.shape[0],
                                               self.embedding_matrix.shape[1],
                                               trainable=True, l2_reg=0)(input1)
        amplitude_encoded1 = amplitude_embedding_layer(np.transpose(self.embedding_matrix),
                                                       self.seq_length,
                                                       trainable=True,
                                                       l2_reg=0.0000005)(input1)

        [seq_embedding_real1, seq_embedding_imag1] = ComplexMultiply()([phase_encoded1, amplitude_encoded1])
        [sentence_embedding_real1, sentence_embedding_imag1] = ComplexMixture()(
            [seq_embedding_real1, seq_embedding_imag1, weight1])

        weight2 = Activation('softmax')(self.weight_embedding(input2))
        phase_encoded2 = phase_embedding_layer(self.seq_length, self.embedding_matrix.shape[0],
                                               self.embedding_matrix.shape[1],
                                               trainable=True, l2_reg=0)(input2)
        amplitude_encoded2 = amplitude_embedding_layer(np.transpose(self.embedding_matrix),
                                                       self.seq_length,
                                                       trainable=True,
                                                       l2_reg=0.0000005)(input1)

        [seq_embedding_real2, seq_embedding_imag2] = ComplexMultiply()([phase_encoded2, amplitude_encoded2])
        [sentence_embedding_real2, sentence_embedding_imag2] = ComplexMixture()([seq_embedding_real2, seq_embedding_imag2, weight2])

        seq_embedding_real = Concatenate()([sentence_embedding_real1, sentence_embedding_real2])
        seq_embedding_imag = Concatenate()([sentence_embedding_imag1, sentence_embedding_imag2])
        [sentence_embedding_real, sentence_embedding_imag] = ComplexMixture(average_weights=True)(
            [seq_embedding_real, seq_embedding_imag])

        probs = ComplexMeasurement(units=30)([sentence_embedding_real, sentence_embedding_imag])
        output = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0))(probs)
        model = Model([input1, input2], output)
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001, decay=0.0), metrics=['accuracy'])
        return model

    def save(self, file):
        self.model.save(file)

    def load(self, file):
        self.model.load_weights(file)
        # json_string = self.model.to_json()
        # self.model = model_from_json(json_string)

    def train(self, train_texts, train_labels, batch_size, epochs, callbacks, validation_split):
        x_train = split_texts(train_texts, self.CLAUSE_NUM, self.seq_length, self.tokenizer)
        y = np.array(train_labels)
        x = []
        for i in range(self.CLAUSE_NUM):
            x.append(x_train[i])
        self.model.summary()
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs,
                       callbacks=callbacks,
                       validation_split=validation_split)

    def evaluate(self, test_texts, test_labels):
        x_test = split_texts(test_texts, self.CLAUSE_NUM, self.seq_length, self.tokenizer)
        y = np.array(test_labels)
        x = []
        for i in range(self.CLAUSE_NUM):
            x.append(x_test[i])
        scores = self.model.evaluate(x, y)
        print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
        return scores

    def predict(self, texts, batch_size=None, verbose=0, steps=None):
        x_test = split_texts(texts, self.CLAUSE_NUM, self.seq_length, self.tokenizer)
        x = []
        for i in range(self.CLAUSE_NUM):
            x.append(x_test[i])
        x = pad_sequences(x, maxlen=self.seq_length, dtype='int32', padding='post', truncating='post')
        return self.model.predict(x, batch_size=batch_size, verbose=verbose, steps=steps)

    def predict_classes(self, texts):
        prediction = self.predict(texts)
        classes = np.argmax(prediction, axis=1)
        return classes

    def predict_prob(self, texts):
        prob = self.predict(texts).squeeze()
        return prob
