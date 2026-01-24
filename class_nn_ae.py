# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.initializers import he_normal
import matplotlib.pyplot as plt
from aux_loss_functions import mean_sum_square_error_loss, mean_sum_binary_crossentropy_loss

###############
# Random seed #
###############


# seed = 2023
# tf.random.set_seed(seed)
# tf.keras.utils.set_random_seed(seed)

############################
# Auxiliary - Denoising AE #
############################


class NoiseLayer(tf.keras.layers.Layer):

    def __init__(self, mean, std):
        super(NoiseLayer, self).__init__()
        self.mean = mean
        self.std = std

    def call(self, input_x):
        mean = self.mean
        std = self.std

        return input_x + tf.random.normal(shape=tf.shape(input_x), mean=mean, stddev=std)

############
# AE class #
############


class AE(tf.keras.Model):

    ###########################################################################################
    #                                          API                                            #
    ###########################################################################################

    ###############
    # Constructor #
    ###############

    # def __init__(self, ae_params):
    def __init__(self,
                 layer_dims,  # shallow: [n_input, n_bottleneck], deep: [n_input, n_encoder_h1,  .., n_bottleneck]
                 learning_rate,
                 num_epochs,
                 minibatch_size,
                 loss_function,  #  'square_error', 'binary_crossentropy'
                 noise_gaussian_avg=0.0,
                 noise_gaussian_std=0.0,
                 l1_activations=0.0,
                 reg_l2=0.0,
                 seed=0
                 ):
        super(AE, self).__init__()
        ##############
        # Vanilla AE #
        ##############
        # seed
        self.seed = seed
        # parameters
        self.layer_dims = layer_dims  # [n_input, n_encoder_h1,  .., n_bottleneck]
        self.rev_layer_dims = [i for i in reversed(self.layer_dims)]
        self.learning_rate = learning_rate
        self.reg_l2 = reg_l2
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size

        # Loss function
        self.loss_function_name = loss_function
        if self.loss_function_name == 'square_error':
            self.output_activation = 'linear'
            loss_function = mean_sum_square_error_loss
        elif self.loss_function_name == 'binary_crossentropy':
            self.output_activation = 'sigmoid'
            loss_function = mean_sum_binary_crossentropy_loss
        else:
            raise Exception('AE() constructor: Wrong loss function')

        # fixed parameters (also, ADAM later)
        self.hidden_activation = 'relu'
        self.weight_init = he_normal(seed=self.seed)

        #############
        # Sparse AE #
        #############

        self.reg_l1 = l1_activations  # NOTE: L1 to activations (not to weights)

        ################
        # Denoising AE #
        ################

        self.noise_gaussian_avg = noise_gaussian_avg
        self.noise_gaussian_std = noise_gaussian_std

        #########
        # Model #
        #########

        # self.autoencoder, self.encoder, self.decoder = self.create_deep_autoencoder()
        self.autoencoder, self.encoder = self.create_deep_autoencoder()

        self.autoencoder.compile(
            optimizer=Adam(lr=self.learning_rate),
            loss=loss_function
        )

    #########
    # Train #
    #########

    def train(self, data, validation_data=None, flag_shuffle=True, flag_plot=False, verbose=0):
        history = self.autoencoder.fit(
            x=data,
            y=data,
            epochs=self.num_epochs,
            batch_size=self.minibatch_size,
            validation_data=validation_data,
            shuffle=flag_shuffle,
            verbose=verbose  # 0: off, 1: full, 2: brief
        )

        if flag_plot:
            plt.plot(history.history['loss'])
            if validation_data is not None:
                plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Test'], loc='upper left')
            plt.show()

    ###########
    # Predict #
    ###########

    # def prediction(self, data):
    #     encodings = self.encoder.predict(data)
    #     reconstructions = self.decoder.predict(encodings)
    #
    #     return encodings, reconstructions
    def prediction(self, data):
        return self.autoencoder.predict(data)

    def prediction_encoded(self, data):
        return self.encoder.predict(data)


    ###########################################################################################
    #                                    Internal functions                                   #
    ###########################################################################################

    ###############
    # Autoencoder #
    ###############

    def create_deep_autoencoder(self):
        n_input = self.layer_dims[0]
        n_encoding = self.layer_dims[-1]

        # input layer
        x_input = Input(shape=(n_input,))

        # noisy input
        new_x_input = NoiseLayer(mean=self.noise_gaussian_avg, std=self.noise_gaussian_std)(x_input)

        # encoder network
        x = Dense(
            units=self.layer_dims[1],
            activation=self.hidden_activation,
            kernel_initializer=self.weight_init,
            kernel_regularizer=l2(self.reg_l2),
            activity_regularizer=l1(self.reg_l1)
        )(new_x_input)

        for n_dim in self.layer_dims[2:-1]:
            x = Dense(
                units=n_dim,
                activation=self.hidden_activation,
                kernel_initializer=self.weight_init,
                kernel_regularizer=l2(self.reg_l2),
                activity_regularizer=l1(self.reg_l1)
            )(x)

        # bottleneck layer
        encoded = Dense(
            units=n_encoding,
            activation=self.hidden_activation,
            kernel_initializer=self.weight_init,
            kernel_regularizer=l2(self.reg_l2),
            activity_regularizer=l1(self.reg_l1)
        )(x)

        # decoder network
        x = Dense(
            units=self.rev_layer_dims[1],
            activation=self.hidden_activation,
            kernel_initializer=self.weight_init,
            kernel_regularizer=l2(self.reg_l2),
            activity_regularizer=l1(self.reg_l1)
        )(encoded)

        for n_dim in self.rev_layer_dims[2:-1]:
            x = Dense(
                units=n_dim,
                activation=self.hidden_activation,
                kernel_initializer=self.weight_init,
                kernel_regularizer=l2(self.reg_l2),
                activity_regularizer=l1(self.reg_l1)
            )(x)

        # output layer
        decoded = Dense(
            units=n_input,
            activation=self.output_activation,
            kernel_initializer=None,
            activity_regularizer=None
        )(x)

        # networks
        encoder = Model(x_input, encoded)
        # decoder = Model(encoded, decoded)
        autoencoder = Model(x_input, decoded)

        # return autoencoder, encoder, decoder
        return autoencoder, encoder
