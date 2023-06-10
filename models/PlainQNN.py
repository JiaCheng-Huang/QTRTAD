# -*- coding: utf-8 -*-
from keras import Model
from keras.layers import *
from keras.optimizers import RMSprop

from layers.embedding import phase_embedding_layer, amplitude_embedding_layer
from layers.measurement import ComplexMeasurement
from layers.mixture import ComplexMixture
from layers.multiply import ComplexMultiply
from models.BaseModel import BaseModel


class PlainQNN(BaseModel):
    def __init__(self, embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE):
        self.weight_embedding = Embedding(embedding_matrix.shape[0], 1, trainable=True)
        super().__init__(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)

    def build_model(self):
        input = Input(shape=(self.seq_length,), dtype='int32')
        weight = Activation('softmax')(self.weight_embedding(input))
        phase_encoded = phase_embedding_layer(self.seq_length, self.embedding_matrix.shape[0],
                                              self.embedding_matrix.shape[1],
                                              trainable=True, l2_reg=0)(input)
        amplitude_encoded = amplitude_embedding_layer(np.transpose(self.embedding_matrix),
                                                      self.seq_length,
                                                      trainable=True,
                                                      l2_reg=0.0000005)(input)
        seq_embedding_real, seq_embedding_imag = ComplexMultiply()([phase_encoded, amplitude_encoded])
        sentence_embedding_real, sentence_embedding_imag = ComplexMixture()(
            [seq_embedding_real, seq_embedding_imag, weight])

        probs = ComplexMeasurement(units=30)([sentence_embedding_real, sentence_embedding_imag])

        output = Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(0))(probs)
        model = Model(input, output)
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001, decay=0.0), metrics=['accuracy'])
        return model

    def save(self, file):
        self.model.save(file)

    def load(self, file):
        self.model.load_weights(file)