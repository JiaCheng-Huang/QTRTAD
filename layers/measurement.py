import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras.models import Model, Input
from keras.constraints import unit_norm
from keras.initializers import Orthogonal


class ComplexMeasurement(Layer):

    def __init__(self, units=5, trainable=True, **kwargs):
        self.units = units
        self.trainable = trainable
        super(ComplexMeasurement, self).__init__(**kwargs)
        self.measurement_constrain = unit_norm(axis=(1, 2))
        self.measurement_initalizer = Orthogonal(gain=1.0, seed=None)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(input_shape) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(input_shape)) + ' inputs.')
        self.dim = input_shape[0][-1]
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.units, self.dim, 2),
                                      constraint=self.measurement_constrain,
                                      initializer=self.measurement_initalizer,
                                      trainable=self.trainable)

        super(ComplexMeasurement, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.')

        if len(inputs) != 2:
            raise ValueError('This layer should be called '
                             'on a list of 2 inputs.'
                             'Got ' + str(len(inputs)) + ' inputs.')

        kernel_real = self.kernel[:, :, 0]
        kernel_imag = self.kernel[:, :, 1]

        input_real = inputs[0]
        input_imag = inputs[1]
        # print(input_real.shape)  #(?, 50, 50)
        # print(input_imag.shape)
        kernel_r = K.batch_dot(K.expand_dims(kernel_real, 1), K.expand_dims(kernel_real, 2), axes=(1, 2)) - K.batch_dot(
            K.expand_dims(kernel_imag, 1), K.expand_dims(kernel_imag, 2), axes=(1, 2))
        # print(kernel_r.shape)
        kernel_i = K.batch_dot(K.expand_dims(kernel_imag, 1), K.expand_dims(kernel_real, 2), axes=(1, 2)) + K.batch_dot(
            K.expand_dims(kernel_real, 1), K.expand_dims(kernel_imag, 2), axes=(1, 2))

        kernel_r = K.reshape(kernel_r, shape=(self.units, self.dim * self.dim))
        kernel_i = K.reshape(kernel_i, shape=(self.units, self.dim * self.dim))
        # print(kernel_r.shape) (3, 25)
        # print(input_real.shape[1:-2]) (4, )
        new_shape = [-1]
        for i in input_real.shape[1:-2]:
            new_shape.append(int(i))
        # print(self.dim) 5
        new_shape.append(self.dim * self.dim)
        # print(new_shape) [-1, 4, 25]
        input_real = K.reshape(input_real, shape=tuple(new_shape))
        input_imag = K.reshape(input_imag, shape=tuple(new_shape))

        output = K.dot(input_real, K.transpose(kernel_r)) - K.dot(input_imag, K.transpose(kernel_i))

        return (output)

    def compute_output_shape(self, input_shape):
        output_shape = [None]
        for i in input_shape[0][1:-2]:
            output_shape.append(i)
        output_shape.append(self.units)
        #        output_shape = [input_shape[0][0:-3],self.units]

        #        print('Input shape of measurment layer:{}'.format(input_shape))
        return [tuple(output_shape)]


def main():
    input_1 = Input(shape=(4, 5, 5), dtype='float')
    input_2 = Input(shape=(4, 5, 5), dtype='float')
    output = ComplexMeasurement(3)([input_1, input_2])

    model = Model([input_1, input_2], output)
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    # model.summary()

    weights = model.get_weights()
    x_1 = np.random.random((5, 4, 5, 5))
    x_2 = np.random.random((5, 4, 5, 5))
    output = model.predict([x_1, x_2])
    for i in range(5):
        xy = x_1[i] + 1j * x_2[i]
        for j in range(3):
            m = weights[0][j, :, 0] + 1j * weights[0][j, :, 1]


#            print(np.matmul(xy[0] ,np.outer(m,m)))
#            result = np.absolute(np.trace(np.matmul(xy ,np.outer(m,m))))
#             print(np.trace(np.matmul(xy[0] ,np.outer(m,m))))


# complex_array = np.random.random((3,5,2))

# norm_2 = np.linalg.norm(complex_array, axis = (1,2))

# for i in range(complex_array.shape[0]):
#     complex_array[i] = complex_array[i]/norm_2[i]
# # complex_array()= complex_array / norm_2
# # x_2 = np.random.random((3,5))

# x = complex_array[:,:,0]
# x_2 = complex_array[:,:,1]
# y = np.array([[1],[1],[0]])
# print(x)
# print(x_2)

# for i in range(1000):
#     model.fit([x,x_2],y)
# # print(model.get_weights())
#     output = model.predict([x,x_2])
#     print(output)
# # print(output)

if __name__ == '__main__':
    main()
