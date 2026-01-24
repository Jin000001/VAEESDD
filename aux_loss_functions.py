# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


###########
# KL loss #
###########


def calc_kl_loss(z_log_var, z_mean, beta):
    return tf.reduce_mean(
        calc_kl_loss_vec(z_log_var, z_mean, beta)  # NOTE: includes beta
    )


def calc_kl_loss_vec(z_log_var, z_mean, beta):
    return beta * tf.reduce_sum(
            -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
            axis=1
    )

#######################
# Reconstruction loss #
#######################


def calc_reconstruction_loss(data, reconstruction, loss_fun):
    ae_loss = None
    if loss_fun == 'square_error':
        ae_loss = mean_sum_square_error_loss(data, reconstruction)
    elif loss_fun == 'binary_crossentropy':
        ae_loss = mean_sum_binary_crossentropy_loss(data, reconstruction)

    return ae_loss


def calc_reconstruction_loss_vec(data, reconstruction, loss_fun):
    ae_loss_vec = None
    if loss_fun == 'square_error':
        ae_loss_vec = sum_square_error_loss(data, reconstruction)
    elif loss_fun == 'binary_crossentropy':
        ae_loss_vec = sum_binary_crossentropy_loss(data, reconstruction)

    return ae_loss_vec

def calc_reconstruction_loss_vec_cae(data, reconstruction, loss_fun,lam,model,encoded):
    ae_loss_vec = None
    if loss_fun == 'square_error':
        ae_loss_vec = sum_square_error_loss_cae(data, reconstruction,lam,model,encoded)
    elif loss_fun == 'binary_crossentropy':
        ae_loss_vec = sum_binary_crossentropy_loss_cae(data, reconstruction,lam,model,encoded)

    return ae_loss_vec

# NOTE: for all the following (see 3D losses below)
# - data is of shape (#, n)
# - if data is of shape (#, n, n, #channels) then use tf.reduce_sum(axis=(1,2))



# #to try: use the tf default which can check the dimension automatically
# def loss_mse(y_true, y_pred):
#     return tf.reduce_mean( tf.losses.mse(y_true, y_pred) )


def mean_sum_square_error_loss(y_true, y_pred):
    return tf.reduce_mean(
        sum_square_error_loss(y_true, y_pred)
    )






def sum_square_error_loss(y_true, y_pred):
    return 0.5 * tf.reduce_sum(
            tf.square(tf.subtract(y_true, y_pred)),
            axis=1
        )

def sum_square_error_loss_cae(y_true, y_pred,lamda,model,encoded):
    rl=sum_square_error_loss(y_true, y_pred)
    cl=contractive_loss(model,lamda,encoded)
    rl = tf.cast(rl, dtype=tf.float64)
    cl = tf.cast(cl, dtype=tf.float64)
    tl = rl + cl
    return tl


def mean_sum_binary_crossentropy_loss(y_true, y_pred):
    return tf.reduce_mean(
        sum_binary_crossentropy_loss(y_true, y_pred)
    )


###mean_sum_binary_crossentropy_loss plus contractive_loss



def sum_binary_crossentropy_loss(y_true, y_pred):
    bce = y_true * tf.math.log(y_pred + tf.keras.backend.epsilon())
    bce += (1 - y_true) * tf.math.log(1 - y_pred + tf.keras.backend.epsilon())
    return tf.reduce_sum(-bce, axis=1)

def sum_binary_crossentropy_loss_cae(y_true, y_pred,lamda,model,encoded):
    rl=sum_binary_crossentropy_loss(y_true, y_pred)
    print('value rl',rl)
    cl=contractive_loss(model,lamda,encoded)
    print('value cl', cl)
    # print(np.shape(cl))
    tl=rl+cl

    return tl


# def contractive_loss(model,lamda):
#     W = K.constant(model.autoencoder.get_layer('encoded').get_weights()[0])
#     W = K.transpose(W)
#     h = model.autoencoder.get_layer('encoded').output
#     dh = h * (1 - h)
#     contractive=lamda *tf.reduce_sum(tf.linalg.matmul(dh ** 2, tf.square(W)), axis=1)
#     return contractive #returns total loss calculated

def contractive_loss(model,lam,encoded):

    W = tf.Variable(model.autoencoder.get_layer('encoded').get_weights()[0])
    h=tf.convert_to_tensor(encoded)
    dh = h * (1 - h)  # N_batch x N_hidden
    W = tf.transpose(W)
    contractive = lam * tf.reduce_sum(tf.linalg.matmul(dh ** 2, tf.square(W)), axis=1)

    return contractive
##########################
# Reconstruction loss 3D #
##########################
# NOTE: see another note earlier


def mean_sum_square_error_loss_3d(y_true, y_pred):
    return tf.reduce_mean(
        sum_square_error_loss_3d(y_true, y_pred)
    )



def sum_square_error_loss_3d(y_true, y_pred):
    return 0.5 * tf.reduce_sum(
            tf.square(tf.subtract(y_true, y_pred)),
            axis=(1, 2)
        )


