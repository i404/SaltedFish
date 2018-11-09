from keras import Input, regularizers, Model
from keras.layers import Embedding, Flatten, Convolution1D, Dense, \
    BatchNormalization, Dropout
from keras.layers.merge import concatenate
from keras.initializers import RandomNormal
from keras import backend as K
import numpy as np

from reprocess import reshape_2d_feature_for_1d_cnn
from models import BasicModel


def embedding_init(shape, name=None):
    return RandomNormal(mean=0.01, stddev=0.01, seed=None)(shape)


class CnnWithEmbedding(BasicModel):

    def __init__(
            self, stock_num=3600, embedding_dim=100,
            cnn_input_shape=None, cnn_filter_nums=None,
            cnn_kernel_size=3, cnn_feature_num=100,
            dense_layer_nodes=None, dense_layer_dropout=None,
            *args, **kwargs):
        if dense_layer_dropout is None:
            dense_layer_dropout = [0.1, 0.3, 0.5]
        self.dense_layer_dropout = dense_layer_dropout
        if cnn_filter_nums is None:
            cnn_filter_nums = [32, 16, 8]
        self.cnn_filter_nums = cnn_filter_nums
        if dense_layer_nodes is None:
            dense_layer_nodes = [32, 16, 8]
        self.dense_layer_nodes = dense_layer_nodes

        self.stock_num = stock_num
        self.embedding_dim = embedding_dim
        self.cnn_input_shape = cnn_input_shape
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_feature_num = cnn_feature_num
        super().__init__(*args, **kwargs)

    def _reshape_input(self, raw_features):
        seq_features = np.array([f[0] for f in raw_features])
        stock_ids = np.array([f[1] for f in raw_features])

        shape, feature = reshape_2d_feature_for_1d_cnn(seq_features)

        self.cnn_input_shape = shape
        return [feature, stock_ids]

    def _create(self):

        # embedding
        stock_id = Input(shape=(1,), dtype="int32", name="stock_id")
        stock_embedding = Embedding(
            input_dim=self.stock_num,
            output_dim=self.embedding_dim,
            name="stock_embedding",
            embeddings_initializer=embedding_init,
            embeddings_regularizer=regularizers.l1(0.0001),
            input_length=1)(stock_id)

        stock_latent = Flatten()(stock_embedding)

        # CNN
        seq_input = Input(shape=self.cnn_input_shape,
                          dtype="float32",
                          name="stock_changes")
        cnn_output = seq_input
        for filter_num in self.cnn_filter_nums:
            cnn_layer = Convolution1D(
                filters=filter_num,
                kernel_size=self.cnn_kernel_size,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l1(0.0001),
                name=f"cnn_with_{filter_num}filters")
            cnn_output = cnn_layer(cnn_output)
            # cnn_output = BatchNormalization()(cnn_layer(cnn_output))

        flatten_cnn_output = Flatten()(cnn_output)
        seq_latent = Dense(
            self.cnn_feature_num, activation="relu",
            name="seq_latent_dense")(flatten_cnn_output)

        # MLP
        latents = concatenate([stock_latent, seq_latent])
        for i in range(0, len(self.dense_layer_nodes)):
            nodes = self.dense_layer_nodes[i]
            dropout_frac = self.dense_layer_dropout[i]
            dense_latents = Dense(
                nodes,
                activation="relu",
                kernel_regularizer=regularizers.l1(0.0001),
                name=f"dense_with_{nodes}nodes")(latents)
            dropout_latents = Dropout(dropout_frac)(dense_latents)
            latents = BatchNormalization()(dropout_latents)

        prediction = Dense(
            1,
            activation='sigmoid',
            name="prediction")(latents)

        model = Model(inputs=[seq_input, stock_id], outputs=prediction)

        model.compile(
            optimizer="adam",
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model
