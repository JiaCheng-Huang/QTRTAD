from keras import Sequential
from keras.layers import *
from keras.optimizers import *

from models.BaseModel import BaseModel


class TextCNN(BaseModel):
    def __init__(self, embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE):
        super().__init__(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)

    def build_model(self):
        # 定义输入层
        filters = 250
        kernel_size = 3
        hidden_dims = 250
        model = Sequential()

        model.add(Embedding(  # Layer 0, Start
            input_dim=self.VOCAB_SIZE + 1,  # Size to dictionary, has to be input + 1
            output_dim=self.embedding_dim,  # Dimensions to generate
            weights=[self.embedding_matrix],  # Initialize word weights
            input_length=self.seq_length,
            name="embedding_layer",
            trainable=False))
        model.add(Dropout(0.2))

        model.add(Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1))
        model.add(GlobalMaxPooling1D())

        model.add(Dense(hidden_dims))
        model.add(Dropout(0.2))
        model.add(Activation('relu'))
        model.add(Dense(2, activation='softmax', name="dense_one", kernel_regularizer=regularizers.l2(0)))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])
        return model
    def get_model(self):
        return self.model
