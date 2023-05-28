from keras.engine.saving import load_model
from keras.layers import *
from keras_preprocessing.sequence import pad_sequences


class BaseModel:
    def __init__(self, embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE):
        self.embedding_matrix = embedding_matrix
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim
        self.VOCAB_SIZE = VOCAB_SIZE
        self.model = self.build_model()

    def train(self, train_texts, train_labels, batch_size, epochs, callbacks, validation_split):
        x = self.tokenizer.texts_to_sequences(train_texts)
        x = pad_sequences(x, maxlen=self.seq_length, dtype='int32', padding='post', truncating='post')
        y = np.array(train_labels)
        self.model.summary()
        self.model.fit(x, y, batch_size=batch_size, epochs=epochs,
                       callbacks=callbacks,
                       validation_split=validation_split)

    def save(self, file):
        self.model.save(file)

    def load(self, file):
        self.model = load_model(file)

    def evaluate(self, test_texts, test_labels):
        x = self.tokenizer.texts_to_sequences(test_texts)
        x = pad_sequences(x, maxlen=self.seq_length, dtype='int32', padding='post', truncating='post')
        y = np.array(test_labels)
        scores = self.model.evaluate(x, y)
        print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
        return scores

    def predict(self, texts, batch_size=None, verbose=0, steps=None):
        x = self.tokenizer.texts_to_sequences(texts)
        x = pad_sequences(x, maxlen=self.seq_length, dtype='int32', padding='post', truncating='post')
        return self.model.predict(x, batch_size=batch_size, verbose=verbose, steps=steps)

    def predict_classes(self, texts):
        prediction = self.predict(texts)
        classes = np.argmax(prediction, axis=1)
        return classes

    def predict_prob(self, texts):
        prob = self.predict(texts).squeeze()
        return prob

    def build_model(self):
        pass
