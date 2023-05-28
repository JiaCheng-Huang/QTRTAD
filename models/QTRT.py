# -*- coding: utf-8 -*-
import nltk
import numpy as np
from keras import Model
from keras.engine.saving import model_from_json
from keras.layers import *
from keras.optimizers import *

from layer.measurement import MeasurementLayer
from layer.mixture import MixtureLayer
from layer.product import ProductLayer
from utils import split_texts

nltk.data.path.append("nltk_data")


class QTRTnn:
    def __init__(self, embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE, CLAUSE_NUM):
        self.embedding_matrix = embedding_matrix
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.VOCAB_SIZE = VOCAB_SIZE
        self.CLAUSE_NUM = CLAUSE_NUM
        self.model = self.build_model()

    def build_model(self):
        input_layers = []
        for i in range(self.CLAUSE_NUM):
            input_layers.append(Input(shape=(self.seq_length,), dtype='int32'))

        embs = []
        for i in range(self.CLAUSE_NUM):
            embs.append(Embedding(
                input_dim=self.VOCAB_SIZE + 1,
                output_dim=self.embedding_dim,
                weights=[self.embedding_matrix],
                input_length=self.seq_length,
                trainable=False
            )(input_layers[i]))

        products = []
        for i in range(self.CLAUSE_NUM):
            products.append(ProductLayer()(embs[i]))

        mixtures = []
        for i in range(self.CLAUSE_NUM):
            mixtures.append(MixtureLayer()(products[i]))

        mixture = Add()(mixtures)

        prob = MeasurementLayer(2)(mixture)


        output_layer = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0))(prob)
        model = Model(input_layers, output_layer)
        # model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])
        # model.compile(loss='categorical_hinge', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
        return model

    def train(self, train_texts, train_labels, batch_size, epochs, callbacks, validation_split=0):
        x_train = split_texts(train_texts, self.CLAUSE_NUM, self.seq_length, self.tokenizer)
        y = np.array(train_labels)
        x = []
        for i in range(self.CLAUSE_NUM):
            x.append(x_train[i])
        self.model.summary()
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs,
                       callbacks=callbacks,
                       validation_split=validation_split)

    def save(self, file):
        self.model.save_weights(file)

    def load(self, file):
        self.model.load_weights(file)
        # json_string = self.model.to_json()
        # self.model = model_from_json(json_string)

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
        return self.model.predict(x, batch_size=batch_size, verbose=verbose, steps=steps)

    def predict_classes(self, texts):
        prediction = self.predict(texts)
        classes = np.argmax(prediction, axis=1)
        return classes

    def predict_prob(self, texts):
        prob = self.predict(texts).squeeze()
        return prob
