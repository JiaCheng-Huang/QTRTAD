from math import sqrt

from keras import Sequential, Model
from keras.layers import *
import tensorflow as tf
from keras.optimizers import RMSprop, Adam
from tensorflow.keras import layers

from models.BaseModel import BaseModel


class Transformer(BaseModel):
    def __init__(self, embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE):
        super().__init__(embedding_matrix, tokenizer, seq_length, embedding_dim, VOCAB_SIZE)

    def load(self, file):
        self.model.load_weights(file)

    def transformer_block(self, x, prefix):
        O_seq = MultiHeadAttention(head_num=2, name=f'{prefix}_att1')(x)  # bs,words_len,dim
        O_seq_Add1 = Add(name=f'{prefix}_add1')([x, O_seq])
        O_seq_LN1 = LayerNorm(name=f'{prefix}_LN1')(O_seq_Add1)  # X = LayerNorm(X + multihead(X))
        O_seq_fc1 = Dense(self.embedding_dim * 4, activation='relu', name=f'{prefix}_fc1')(O_seq_LN1)  # FFN
        O_seq_fc2 = Dense(self.embedding_dim, name=f'{prefix}_fc2')(O_seq_fc1)
        O_seq_Add2 = Add(name=f'{prefix}_add2')([O_seq_LN1, O_seq_fc2])  #
        O_seq_Add2 = add([O_seq_LN1, O_seq_fc2])
        O_seq_LN2 = LayerNorm(name=f'{prefix}_LN2')(O_seq_Add2)
        return O_seq_LN2

    def build_model(self):
        words = Input(shape=(self.seq_length,), name='inputs', dtype='int32')
        embeddings = Embedding(*self.embedding_matrix.shape, weights=[self.embedding_matrix], trainable=True)(words)
        embeddings = Position_Embedding()(embeddings)  # 增加Position_Embedding能轻微提高准确率
        embeddings = Dropout(0.1)(embeddings)

        # def transformer_block(x,prefix):
        seq_len = K.shape(words)[1]
        #     model_dim = K.int_shape(embeddings)[-1]

        O_seq1 = self.transformer_block(embeddings, prefix='t1')
        O_seq2 = self.transformer_block(O_seq1, prefix='t2')
        O_seq3 = self.transformer_block(O_seq2, prefix='t3')
        O_seq4 = self.transformer_block(O_seq3, prefix='t4')
        O_seq5 = self.transformer_block(O_seq4, prefix='t5')
        O_seq6 = self.transformer_block(O_seq5, prefix='t6')
        #     O_seq7 = transformer_block(O_seq6,prefix='t7')
        #     O_seq8 = transformer_block(O_seq7,prefix='t8')

        O_seq = Add()([O_seq4, O_seq5, O_seq6])  ###后面这块是自由发挥的
        O_seq = GlobalAveragePooling1D()(O_seq)
        O_seq = Dropout(0.1)(O_seq)

        # 下面的这块原文用了warmup，我们不用了。

        result = Dense(2, activation='softmax', name='outputs')(O_seq)
        model = Model(inputs=words, outputs=result)
        opt = Adam(lr=5e-5)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])
        return model


class MultiHeadAttention(Layer):
    """Multi-head attention layer.
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.
        :param head_num: Number of heads.
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """
        self.supports_masking = True
        self.head_num = head_num
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.history_only = history_only

        self.Wq = self.Wk = self.Wv = self.Wo = None
        self.bq = self.bk = self.bv = self.bo = None

        self.intensity = self.attention = None
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': self.head_num,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
            return q[:-1] + (v[-1],)
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = int(v[-1])
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        self.Wq = self.add_weight(
            shape=(int(q[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(int(k[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(int(v[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHeadAttention, self).build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num):
        # split to head num
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        ##为了方便scaled dot attention 计算（输入是bs, seq_len,head_dim）,这里做了transpose和reshape
        x = K.permute_dimensions(x, [0, 2, 1, 3])  # transpose,把并行计算的head_num维度提到前面
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))  # reshape,因为bs轴在scaled dot里面不参与计算

    @staticmethod
    def _reshape_attention_from_batches(x, head_num):  ##attention得分矩阵的反向恢复
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        return K.permute_dimensions(x, [0, 2, 1, 3])

    @staticmethod
    def _reshape_from_batches(x, head_num):  # attention后的向量恢复
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[
            2]  # bs*head_num,seq_len,head_dim
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))  # bs,head_num,seq_len,head_dim
        x = K.permute_dimensions(x, [0, 2, 1, 3])  # bs,seq_len,head_num,head_dim
        return K.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))  # bs,seq_len,model_dim

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, head_num, 1])
        return K.reshape(mask, (-1, seq_len))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs  # bs,seq_len,model_dim
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)  # 先做变换再分成8个，和先分成8*64个再做变换，参数量都是一样的512*512
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        scaled_dot_product_attention = ScaledDotProductAttention(
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )
        y = scaled_dot_product_attention(
            inputs=[
                self._reshape_to_batches(q, self.head_num),  # query,bs*numhead,seq_len,dim,head_dim
                self._reshape_to_batches(k, self.head_num),  # key
                self._reshape_to_batches(v, self.head_num),  # value
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        #       相似度矩阵
        #         self.intensity = self._reshape_attention_from_batches(scaled_dot_product_attention.intensity, self.head_num)
        #         self.attention = self._reshape_attention_from_batches(scaled_dot_product_attention.attention, self.head_num)
        y = self._reshape_from_batches(y, self.head_num)  # 合并
        y = K.dot(y, self.Wo)  # 最终输出
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)

        # Add shape information to tensor
        input_shape = [K.int_shape(q), K.int_shape(k), K.int_shape(v)]
        output_shape = self.compute_output_shape(input_shape)
        if output_shape[1] is not None:
            output_shape = (-1,) + output_shape[1:]
            y = K.reshape(y, output_shape)
        return y


class LayerNorm(Layer):
    def __init__(self,
                 center=True,
                 scale=False,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs
                 ):
        super(LayerNorm, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma, self.beta = 0., 0.

    def call(self, inputs, **kwargs):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs


class Position_Embedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):  # 上一层一般就是embedding层，batch_size,seq_len,model_dim
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])  # d_model的长度,比如512
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]  #
        ## K.arange(self.size / 2, dtype='float32' ), 生成0~256，间隔1,即公式中的i
        ## 2*K.arange(self.size / 2, dtype='float32' ), 0~512，间隔2,即公式中的2i, 0,2,4,6……,512，对应的i是0,1,2,3,4,5
        ## 再除以model_dim，按公式取pow
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)  #
        position_j = K.expand_dims(position_j, 0)  # (1,256)
        # 生成位置的序列
        # x[:,:,0]取每个embedding的第一个分量---> bs,seq_len
        # ones_like -->bs,seq_len [[1，1，1，1……],[1,1,1……],……]
        # cumsum ---> bs,seq_len,[[1,2,3,4……],[1,2,3……],……]
        # cumsum-1 ----->bs,seq_len,[[0,1,2,3……],[0,1,2……],……]
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)  # bs,seq_len,1
        position_ij = K.dot(position_i, position_j)  # bs,seq_len,256
        ##经过dot之后,就是pe/10000^(2i/d_model)了
        ##原始的实现稍微有点问题，不应该直接concatenate偶数和奇数，应该交叉concatenate
        position_ij_2i = K.sin(position_ij)[..., tf.newaxis]  # bs,seq_len,model_dim/2,1
        position_ij_2i_1 = K.cos(position_ij)[..., tf.newaxis]  # bs,seq_len,model_dim/2,1
        position_ij = K.concatenate([position_ij_2i, position_ij_2i_1])  # bs,seq_len,model_dim/2,2
        position_ij = K.reshape(position_ij, (batch_size, seq_len, self.size))  # bs,seq_len,model_dim
        # position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)#这个实现没有交叉拼接，前半部分都用的cos，后半部分都用的sin
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class ScaledDotProductAttention(Layer):
    r"""The attention layer that takes three inputs representing queries, keys and values.
    \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 return_attention=False,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.
        :param return_attention: Whether to return attention weights.
        :param history_only: Whether to only use history data.
        :param kwargs: Arguments for parent class.
        """
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only
        self.intensity = self.attention = None

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        feature_dim = K.shape(query)[-1]  # 512
        # query = (bs,seq_len,dim)
        # key = (bs,seq_len,dim)
        # batch_dot后bs,seq_len,seq_len
        e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        if self.history_only:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            indices = K.expand_dims(K.arange(0, key_len), axis=0)
            upper = K.expand_dims(K.arange(0, query_len), axis=-1)
            e -= 10000.0 * K.expand_dims(K.cast(indices > upper, K.floatx()), axis=0)
        if mask is not None:
            e -= 10000.0 * (1.0 - K.cast(K.expand_dims(mask, axis=-2), K.floatx()))
        self.intensity = e
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        self.attention = e / K.sum(e, axis=-1, keepdims=True)
        # self.attention = bs,seq_len,seq_len
        # value = bs,seq_len,dim
        # v = bs,seq_len,dim
        v = K.batch_dot(self.attention, value)
        if self.return_attention:
            return [v, self.attention]
        return v
