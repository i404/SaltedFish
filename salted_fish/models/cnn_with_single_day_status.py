from keras import Input, regularizers, Model
from keras.layers import Embedding, Flatten, Convolution1D, Dense, \
    BatchNormalization, Dropout, MaxPooling1D
from keras.layers.merge import concatenate
from keras.initializers import RandomNormal
from keras import backend as K
import numpy as np
from keras.optimizers import Adam

from reprocess import reshape_2d_feature_for_1d_cnn
from models import BasicModel
from stock_reader import MatrixReaderWithIdAndStatus


def embedding_init(shape, name=None):
    return RandomNormal(mean=0.01, stddev=0.01, seed=None)(shape)


class CnnWithSingleDayStatus(BasicModel):

    def _create_reader(self):
        return MatrixReaderWithIdAndStatus(
            self.data_path, self.index_file, self.sequence_length)

    def __init__(
            self, stock_num=3600, embedding_dim=100,
            cnn_input_shape=None, cnn_filter_nums=None,
            cnn_dropout=None, cnn_kernel_size=3,
            cnn_regularize=None, cnn_feature_num=100,
            dense_layer_nodes=None, dense_layer_dropout=None,
            dense_regularize=None,
            single_day_change_status_embedding_dim=512,
            status_embedding_regularize=5e-5,
            *args, **kwargs):

        super().__init__(*args, **kwargs)

        if cnn_regularize is None:
            cnn_regularize = [5e-5, 5e-5, 5e-5]
        self.cnn_regularize = cnn_regularize
        if cnn_filter_nums is None:
            cnn_filter_nums = [32, 16, 8]
        self.cnn_filter_nums = cnn_filter_nums
        if cnn_dropout is None:
            cnn_dropout = [0.1, 0.1, 0.1]
        self.cnn_dropout = cnn_dropout

        if dense_layer_dropout is None:
            dense_layer_dropout = [0.1, 0.3, 0.5]
        self.dense_layer_dropout = dense_layer_dropout
        if dense_layer_nodes is None:
            dense_layer_nodes = [32, 16, 8]
        self.dense_layer_nodes = dense_layer_nodes
        if dense_regularize is None:
            dense_regularize = [5e-5, 5e-5, 5e-5]
        self.dense_regularize = dense_regularize

        self.stock_num = stock_num
        self.embedding_dim = embedding_dim
        self.cnn_input_shape = cnn_input_shape
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_feature_num = cnn_feature_num

        self.single_day_change_status_embedding_dim = \
            single_day_change_status_embedding_dim
        self.single_day_status_embedding_regularize = \
            status_embedding_regularize

    def _reshape_input(self, raw_features):
        seq_features = np.array([f[0] for f in raw_features])
        # stock_ids = np.array([f[1] for f in raw_features])
        # date_inds = np.array([f[2] for f in raw_features])
        date_inds = np.array([f[1] for f in raw_features])

        shape, feature = reshape_2d_feature_for_1d_cnn(seq_features)

        self.cnn_input_shape = shape
        # return [feature, stock_ids, date_inds]
        return [feature, date_inds]

    def _stock_embedding_part(self):
        """
        embedding latent
        :return: embedding
        """
        stock_id = Input(shape=(1,), dtype="int32", name="stock_id")
        stock_embedding = Embedding(
            input_dim=self.stock_num,
            output_dim=self.embedding_dim,
            name="stock_embedding",
            embeddings_initializer=embedding_init,
            # embeddings_regularizer=regularizers.l1(0.0001),
            input_length=1)(stock_id)
        stock_embedding_flatten = Flatten()(stock_embedding)
        return stock_id, BatchNormalization()(stock_embedding_flatten)

    def _cnn_part(self):
        """
        extract feature from sequence with cnn
        :return: feature from seq
        """
        seq_input = Input(shape=self.cnn_input_shape,
                          dtype="float32",
                          name="stock_changes")
        cnn_output = seq_input
        for i in range(0, len(self.cnn_filter_nums)):
            filter_num = self.cnn_filter_nums[i]
            dropout_frac = self.cnn_dropout[i]
            regularize = self.cnn_regularize[i]
            cnn_layer = Convolution1D(
                filters=filter_num,
                kernel_size=self.cnn_kernel_size,
                padding="same",
                activation="relu",
                kernel_regularizer=regularizers.l1(regularize),
                name=f"cnn_with_{filter_num}filters")
            cnn_output = cnn_layer(cnn_output)
            if dropout_frac > 0:
                cnn_output = Dropout(dropout_frac)(cnn_output)
            cnn_output = BatchNormalization()(cnn_output)

        # pooled_output = MaxPooling1D()(cnn_output)
        flatten_cnn_output = Flatten()(cnn_output)
        # seq_latents = Dense(
        #     self.cnn_feature_num,
        #     activation="relu",
        #     name="seq_latent_dense")(flatten_cnn_output)
        # return seq_input, BatchNormalization()(seq_latents)
        return seq_input, BatchNormalization()(flatten_cnn_output)

    def _single_day_change_status_part(self):
        date_ind_status_dict = self.get_reader().single_day_stock_change_status
        stock_num = len(date_ind_status_dict[0])
        date_num = len(date_ind_status_dict)
        embedding_matrix = np.zeros((date_num, stock_num))
        for ind, status in date_ind_status_dict.items():
            embedding_matrix[ind] = np.asarray(status, dtype="float32")

        date_ind = Input(
            shape=(1,),
            dtype="int32",
            name="date_index")

        single_day_change_status = Embedding(
            input_dim=date_num,
            output_dim=stock_num,
            weights=[embedding_matrix],
            input_length=1,
            trainable=False,
            name="single_day_status_dict")(date_ind)

        embedding_regularize = regularizers.l1(
            self.single_day_status_embedding_regularize)
        single_day_status_latents = Dense(
            self.single_day_change_status_embedding_dim,
            activation="relu",
            kernel_regularizer=embedding_regularize,
            name=f"single_day_status_dense")(single_day_change_status)

        flatten_latents = Flatten()(single_day_status_latents)
        return date_ind, BatchNormalization()(flatten_latents)

    def _create(self):

        # stock_id, stock_latent = self._stock_embedding_part()
        seq_input, seq_latent = self._cnn_part()
        date_ind, date_latents = self._single_day_change_status_part()

        # MLP
        # mlp_inputs = [stock_latent, seq_latent, date_latents]
        mlp_inputs = [seq_latent, date_latents]
        latents = concatenate(mlp_inputs)
        for i in range(0, len(self.dense_layer_nodes)):
            nodes = self.dense_layer_nodes[i]
            dropout_frac = self.dense_layer_dropout[i]
            regularize = self.dense_regularize[i]
            dense_latents = Dense(
                nodes,
                activation="relu",
                kernel_regularizer=regularizers.l1(regularize),
                name=f"dense_with_{nodes}nodes")(latents)
            if dropout_frac > 0:
                dense_latents = Dropout(dropout_frac)(dense_latents)
            latents = BatchNormalization()(dense_latents)

        prediction = Dense(
            1,
            activation='sigmoid',
            name="prediction")(latents)

        # input_lst = [seq_input, stock_id, date_ind]
        input_lst = [seq_input, date_ind]
        model = Model(inputs=input_lst, outputs=prediction)

        return model
