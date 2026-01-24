import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.initializers import he_normal
from aux_loss_functions import calc_kl_loss, calc_reconstruction_loss
import random

##################
# Sampling layer #
##################


class Sampler(Layer):

    def call(self, z_mean, z_log_var, seed):
        batch_size, z_size = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch_size, z_size), seed=seed)

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

#######
# VAE #
#######


class VAE(keras.Model):

    ###########################################################################################
    #                                          API                                            #
    ###########################################################################################

    ###############
    # Constructor #
    ###############

    def __init__(self, layer_dims, learning_rate, num_epochs, batch_size, loss_function, beta, dropout_rate, seed,**kwargs):

        # inheritance
        super(VAE, self).__init__(**kwargs)

        # random seed
        self.seed = seed
        tf.random.set_seed(seed)
        random.seed(seed)

        # dropout
        self.dropout_rate = dropout_rate

        # neural architecture
        self.num_epochs = num_epochs
        self.layer_dims = layer_dims
        self.latent_dim = layer_dims[-1]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_init = he_normal(seed=self.seed)
        self.rev_layer_dims = [i for i in reversed(layer_dims)]

        # loss function and last activation
        self.loss_function_name = loss_function
        if loss_function == 'square_error':
            self.output_activation = 'linear'
        elif loss_function == 'binary_crossentropy':
            self.output_activation = 'sigmoid'
        else:
            raise Exception('VAE() constructor: Wrong loss function')

        # VAE
        self.beta = beta
        self.sampler = Sampler()
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.compile(optimizer=Adam(lr=self.learning_rate))

        # metrics
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.ae_loss_tracker = keras.metrics.Mean(name='ae_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')



    ############
    # Training #
    ############

    def history_train(self, data,validation_data=None,flag_shuffle=True, verbose=2):
        history = self.fit(  # NOTE: This calls the train_step(). In other words, we have customised fit().
            data,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            shuffle=flag_shuffle,
            verbose=verbose,  # 0: off, 1: full, 2: brief


        )
        return history
    # def history_train(self, data, flag_shuffle=True, verbose=2):
    #     history = self.fit(  # NOTE: This calls the train_step(). In other words, we have customised fit().
    #         data,
    #         epochs=self.num_epochs,
    #         batch_size=self.batch_size,
    #         shuffle=flag_shuffle,
    #         verbose=verbose,  # 0: off, 1: full, 2: brief
    #     )
    #     return history
    def train(self, data, flag_shuffle=True, verbose=2):
        self.fit(  # NOTE: This calls the train_step(). In other words, we have customised fit().
            data,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            shuffle=flag_shuffle,
            verbose=verbose,  # 0: off, 1: full, 2: brief

        )

    ##############
    # Prediction #
    ##############

    def prediction(self, data):
        z_mean, z_log_var = self.encoder.predict(data)
        reconstruction = self.decoder.predict(z_mean)

        return z_mean, z_log_var, reconstruction

    ###########################################################################################
    #                                    Internal functions                                   #
    ###########################################################################################

    ###########
    # Metrics #
    ###########

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.ae_loss_tracker, self.kl_loss_tracker]

    ###########
    # Encoder #
    ###########

    def create_encoder(self):
        # input layer
        x_input = keras.Input(shape=(self.layer_dims[0],))

        # intermediate layers
        x = Dense(units=self.layer_dims[1], activation='relu', kernel_initializer=self.weight_init)(x_input)
        for n_dim in self.layer_dims[2:-1]:
            x = Dense(units=n_dim, activation='relu', kernel_initializer=self.weight_init)(x)

        # encoding layer
        z_mean = Dense(self.latent_dim, name="z_mean")(x)
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x)

        # model
        encoder = keras.Model(x_input, [z_mean, z_log_var], name="encoder")
        # encoder.summary()
        return encoder

    ###########
    # Decoder #
    ###########

    def create_decoder(self):
        # input layer
        latent_input = keras.Input(shape=(self.latent_dim,))

        # intermediate layers
        x = Dense(units=self.rev_layer_dims[1], activation='relu', kernel_initializer=self.weight_init)(latent_input)
        for n_dim in self.rev_layer_dims[2:-1]:
            x = Dense(units=n_dim, activation='relu', kernel_initializer=self.weight_init)(x)

        # output layer
        decoded = Dense(units=self.rev_layer_dims[-1], activation=self.output_activation,
                        kernel_initializer=self.weight_init)(x)

        # model
        decoder = keras.Model(latent_input, decoded, name="decoder")
        # decoder.summary()
        return decoder



    #################
    # Training step #
    #################

    def train_step(self, data):  # NOTE: The input argument data is what gets passed to fit()
        with tf.GradientTape() as tape:
            # sample from encoder
            z_mean, z_log_var = self.encoder(data)
            z = self.sampler(z_mean, z_log_var, self.seed)

            # reconstruction from decoder
            reconstruction = self.decoder(z)

            # VAE loss
            ae_loss = calc_reconstruction_loss(data, reconstruction, self.loss_function_name)
            ####to try with code online
            # ae_loss=tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction)))
            kl_loss = calc_kl_loss(z_log_var, z_mean, self.beta)
            total_loss = ae_loss + kl_loss  # NOTE: includes beta factor


        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.ae_loss_tracker.update_state(ae_loss)
        self.kl_loss_tracker.update_state(kl_loss)


        return {
            "total_loss": self.total_loss_tracker.result(),
            "ae_loss": self.ae_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),

        }

    def test_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        z_mean, z_log_var = self.encoder(data)
        z = self.sampler(z_mean, z_log_var, self.seed)
        reconstruction = self.decoder(z)
        ae_loss = calc_reconstruction_loss(data, reconstruction, self.loss_function_name)
        kl_loss = calc_kl_loss(z_log_var, z_mean, self.beta)
        total_loss = ae_loss + kl_loss  # NOTE: includes beta factor
        return {
            "total_loss": total_loss,
            "ae_loss": ae_loss,
            "kl_loss": kl_loss,
        }

    # def call(self, data):
    #     z_mean, z_log_var = self.encoder(data)
    #     z = self.sampler(z_mean, z_log_var, self.seed)
    #     reconstruction = self.decoder(z)
    #     ae_loss = calc_reconstruction_loss(data, reconstruction, self.loss_function_name)
    #     kl_loss = calc_kl_loss(z_log_var, z_mean, self.beta)
    #     total_loss = ae_loss + kl_loss  # NOTE: includes beta factor
    #     self.add_metric(kl_loss, name='kl_loss', aggregation='mean')
    #     self.add_metric(total_loss, name='total_loss', aggregation='mean')
    #     self.add_metric(ae_loss, name='ae_loss', aggregation='mean')
    #     return reconstruction
    # def val_step(self, val_data):  # NOTE: The input argument data is what gets passed to fit()
    #     with tf.GradientTape() as tape:
    #         # sample from encoder
    #         z_mean, z_log_var = self.encoder(val_data)
    #         z = self.sampler(z_mean, z_log_var, self.seed)
    #
    #         # reconstruction from decoder
    #         reconstruction = self.decoder(z)
    #
    #         # VAE loss
    #         val_ae_loss = calc_reconstruction_loss(val_data, reconstruction, self.loss_function_name)
    #         val_kl_loss = calc_kl_loss(z_log_var, z_mean, self.beta)
    #         val_total_loss = val_ae_loss + val_kl_loss  # NOTE: includes beta factor
    #
    #
    #
    #     return {
    #
    #         "val_total_loss": self.val_total_loss,
    #
    #     }


##https://kpj.github.io/stats_ml/Autoencoder.html
    def saveKerasModel_vae(self):
        self.save_weights('mnist_vae.h5', save_format='h5')
        # self.save('mnist_vae')



    def loadModel_vae(self):
        self.load_weights('mnist_vae.h5')
        # self.load_model('mnist_vae.h5')
 # vae_loaded = load_model("test_vae", custom_objects={"vae_loss": vae_loss})








#########
# Tests #
#########
# TODO: delete

# import numpy as np

# (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
# mnist_digits = np.concatenate([x_train, x_test], axis=0)
#
# # mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255
#
# mnist_digits = mnist_digits.reshape(mnist_digits.shape[0], mnist_digits.shape[1] * mnist_digits.shape[2])  # reshape
# mnist_digits = mnist_digits.astype('float32') / 255  # convert to float and rescale
#
# mnist_digits = mnist_digits[:5000, :]

## Display a grid of sampled digits


# import numpy as np
# import matplotlib.pyplot as plt
#
# def plot_latent(encoder, decoder):
#     # display a n*n 2D manifold of digits
#     n = 30
#     digit_size = 28
#     scale = 2.0
#     figsize = 15
#     figure = np.zeros((digit_size * n, digit_size * n))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-scale, scale, n)
#     grid_y = np.linspace(-scale, scale, n)[::-1]
#
#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.array([[xi, yi]])
#             x_decoded = decoder.predict(z_sample)
#             digit = x_decoded[0].reshape(digit_size, digit_size)
#             figure[
#                 i * digit_size : (i + 1) * digit_size,
#                 j * digit_size : (j + 1) * digit_size,
#             ] = digit
#
#     plt.figure(figsize=(figsize, figsize))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure, cmap="Greys_r")
#     plt.show()


# plot_latent(vae.encoder, vae.decoder)


#
## Display how the latent space clusters different digit classes
#


# def plot_label_clusters(encoder, decoder, data, labels):
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _ = encoder.predict(data)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
#     plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.show()
#
#
# (x_train, y_train), _ = keras.datasets.mnist.load_data()
# x_train = np.expand_dims(x_train, -1).astype("float32") / 255
#
# plot_label_clusters(vae.encoder, vae.decoder, x_train, y_train)
