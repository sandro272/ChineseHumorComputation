from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers as initializations
from keras import regularizers, constraints

from keras.layers.merge import add, average, concatenate, maximum, multiply



class Attention_layer(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializations.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(Attention_layer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)

        a = K.exp(uit)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[-1]


class AttentionM(Layer):
    """
    Keras layer to compute an attention vector on an incoming matrix.

    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)

    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)

    # Usage
        enc = LSTM(EMBED_SIZE, return_sequences=True)(...)
        att = AttentionM()(enc)

    """

    def __init__(self, **kwargs):
        super(AttentionM, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS,)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        super(AttentionM, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(et)
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # atx: (BATCH_SIZE, MAX_TIMESTEPS, 1)
        atx = K.expand_dims(at, axis=-1)
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ot = x * atx
        # output: (BATCH_SIZE, EMBED_SIZE)
        return K.sum(ot, axis=1)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(AttentionM, self).get_config()


class AttentionMC(Layer):
    """
    Keras layer to compute an attention vector on an incoming matrix
    using a learned context vector.

    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)

    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)

    # Usage
        enc = Bidirectional(GRU(EMBED_SIZE,return_sequences=True))(...)
        att = AttentionMC()(enc)

    """

    def __init__(self, **kwargs):
        super(AttentionMC, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS, 1)
        # u: (MAX_TIMESTEPS, MAX_TIMESTEPS)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros")
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer="normal")
        super(AttentionMC, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        et = K.squeeze(K.tanh(K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.dot(et, self.u))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        return K.sum(ot, axis=1)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(AttentionMC, self).get_config()


class AttentionMV(Layer):
    """
    Keras layer to compute an attention vector on an incoming matrix
    and a user provided context vector.

    # Input
        enc - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        ctx - 2D Tensor of shape (BATCH_SIZE, EMBED_SIZE) (optional)

    # Output
        2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)

    # Usage
        enc = Bidirectional(GRU(EMBED_SIZE,return_sequences=True))(...)

        # with user supplied vector
        ctx = GlobalAveragePooling1D()(enc)
        att = AttentionMV()([enc, ctx])

    """

    def __init__(self, **kwargs):
        super(AttentionMV, self).__init__(**kwargs)

    def build(self, input_shape):
        assert type(input_shape) is list and len(input_shape) == 2
        # W: (EMBED_SIZE, 1)
        # b: (MAX_TIMESTEPS, 1)
        # U: (EMBED_SIZE, MAX_TIMESTEPS)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[0][-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[0][1], 1),
                                 initializer="zeros")
        self.U = self.add_weight(name="U_{:s}".format(self.name),
                                 shape=(input_shape[0][-1],
                                        input_shape[0][1]),
                                 initializer="normal")
        super(AttentionMV, self).build(input_shape)

    def call(self, xs, mask=None):
        # input: [x, u]
        # x: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # u: (BATCH_SIZE, EMBED_SIZE)
        x, c = xs
        # et: (BATCH_SIZE, MAX_TIMESTEPS)
        et = K.dot(c, self.U) + K.squeeze((K.dot(x, self.W) + self.b), axis=-1)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(et)
        if mask is not None and mask[0] is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        return K.sum(ot, axis=1)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        # output shape: (BATCH_SIZE, EMBED_SIZE)
        return (input_shape[0][0], input_shape[0][-1])

    def get_config(self):
        return super(AttentionMV, self).get_config()


class AttentionMM(Layer):
    """
    Keras layer to compute an attention vector on a pair of incoming
    matrices.

    # Input
        m1 - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        m2 - 3D Tensor of shape (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        merge_mode - one of concat, diff, prod, avg or max.

    # Output
        if merge_mode == "concat":
            2D Tensor of shape (BATCH_SIZE, EMBED_SIZE*2)
        else:
            2D Tensor of shape (BATCH_SIZE, EMBED_SIZE)

    # Usage
        enc1 = LSTM(EMBED_SIZE, return_sequences=True)(...)
        enc2 = LSTM(EMBED_SIZE, return_sequences=True)(...)

        att = AttentionMM("concat")([enc1, enc2])
        att = BatchNormalization()(att)

    """

    def __init__(self, merge_mode, **kwargs):
        self.merge_mode = merge_mode
        assert self.merge_mode in set(["concat", "diff", "prod", "avg", "max"])
        super(AttentionMM, self).__init__(**kwargs)

    def build(self, input_shape):
        assert type(input_shape) is list and len(input_shape) == 2
        assert input_shape[0] == input_shape[1]
        # W1: (EMBED_SIZE, 1)
        # b1: (MAX_TIMESTEPS, 1)
        # W2: (EMBED_SIZE, 1)
        # b2: (MAX_TIMESTEPS, 1)
        # W3: (EMBED_SIZE, EMBED_SIZE)
        # b3: (MAX_TIMESTEPS, EMBED_SIZE)
        # W4: (EMBED_SIZE, EMBED_SIZE)
        # b4: (MAX_TIMESTEPS, EMBED_SIZE)
        # U1: (EMBED_SIZE, MAX_TIMESTEPS)
        # U2: (EMBED_SIZE, MAX_TIMESTEPS)
        self.embed_size = input_shape[0][-1]
        self.max_timesteps = input_shape[0][1]
        self.W1 = self.add_weight(name="W1_{:s}".format(self.name),
                                  shape=(self.embed_size, self.embed_size),
                                  initializer="normal")
        self.b1 = self.add_weight(name="b1_{:s}".format(self.name),
                                  shape=(self.max_timesteps, self.embed_size),
                                  initializer="zeros")
        self.W2 = self.add_weight(name="W2_{:s}".format(self.name),
                                  shape=(self.embed_size, self.embed_size),
                                  initializer="normal")
        self.b2 = self.add_weight(name="b2_{:s}".format(self.name),
                                  shape=(self.max_timesteps, self.embed_size),
                                  initializer="zeros")
        self.U1 = self.add_weight(name="U1_{:s}".format(self.name),
                                  shape=(self.embed_size, self.embed_size),
                                  initializer="normal")
        self.U2 = self.add_weight(name="U2_{:s}".format(self.name),
                                  shape=(self.embed_size, self.embed_size),
                                  initializer="normal")
        self.V1 = self.add_weight(name="V1_{:s}".format(self.name),
                                  shape=(self.embed_size, self.embed_size),
                                  initializer="normal")
        self.V2 = self.add_weight(name="V2_{:s}".format(self.name),
                                  shape=(self.embed_size, self.embed_size),
                                  initializer="normal")
        self.b3 = self.add_weight(name="b3_{:s}".format(self.name),
                                  shape=(self.max_timesteps, self.embed_size),
                                  initializer="zeros")
        self.b4 = self.add_weight(name="b4_{:s}".format(self.name),
                                  shape=(self.max_timesteps, self.embed_size),
                                  initializer="zeros")
        super(AttentionMM, self).build(input_shape)

    def call(self, xs, mask=None):
        assert len(xs) == 2
        # separate out input matrices
        # x1.shape == (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # x2.shape == (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        x1, x2 = xs
        # build alignment matrix
        # e1t, e2t: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, MAX_TIMESTEPS)
        e1t = K.relu(K.dot(x1, self.W1) + self.b1)
        e2t = K.relu(K.dot(x2, self.W2) + self.b2)
        et = K.softmax(K.batch_dot(e2t, e1t, axes=(2, 2)))
        # align inputs
        # a1t, a2t: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        a1t = K.batch_dot(et, x2, axes=(1, 1))
        a2t = K.batch_dot(et, x1, axes=(2, 1))
        # produce aligned outputs
        # o1t, o2t: (BATCH_SIZE, MAX_TIMESTEPS*2, EMBED_SIZE)
        o1t = K.relu(K.dot(x1, self.U1) + K.dot(a1t, self.V1) + self.b3)
        o2t = K.relu(K.dot(x2, self.U2) + K.dot(a2t, self.V2) + self.b4)
        if mask is not None and mask[0] is not None:
            o1t *= K.cast(mask, K.floatx())
        if mask is not None and mask[1] is not None:
            o1t *= K.cast(mask, K.floatx())
        # o1, o2: (BATCH_SIZE, EMBED_SIZE)
        o1 = K.mean(o1t, axis=1)
        o2 = K.mean(o2t, axis=1)
        # merge the attention vectors according to merge_mode
        if self.merge_mode == "concat":
            return concatenate([o1, o2], axis=1)
        elif self.merge_mode == "diff":
            return add([o1, -o2])
        elif self.merge_mode == "prod":
            return multiply([o1, o2])
        elif self.merge_mode == "avg":
            return average([o1, o2])
        else:  # max
            return maximum([o1, o2])

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def compute_output_shape(self, input_shape):
        if self.merge_mode == "concat":
            # output shape: (BATCH_SIZE, EMBED_SIZE*2)
            return (input_shape[0][0], input_shape[0][2] * 2)
        else:
            # output shape: (BATCH_SIZE, EMBED_SIZE)
            return (input_shape[0][0], input_shape[0][2])

    def get_config(self):
        config = {"merge_mode": self.merge_mode}
        base_config = super(AttentionMM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))