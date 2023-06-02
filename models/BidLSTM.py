from keras import Sequential
from keras.layers import *
from keras.optimizers import *

from models.BaseModel import BaseModel


class BidLSTM(BaseModel):
    def __init__(self, embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE):
        super().__init__(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)

    def build_model(self):
        model = Sequential()
        model.add(Embedding(  # Layer 0, Start
            input_dim=self.VOCAB_SIZE + 1,  # Size to dictionary, has to be input + 1
            output_dim=self.embedding_dim,  # Dimensions to generate
            weights=[self.embedding_matrix],  # Initialize word weights
            input_length=self.seq_length,
            name="embedding_layer",
            trainable=False))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])
        return model

    def get_model(self):
        return self.model
