# -*- coding: utf-8 -*-
import os
import time
from sklearn.metrics import f1_score
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from collections import deque
from aux_loss_functions import  calc_reconstruction_loss_vec,calc_kl_loss_vec
from scipy.stats import ranksums,mannwhitneyu
from scipy import stats
from data.mnist.mnist_0123 import preprocess_mnist,pretrain_mnist_model
import matplotlib.pyplot as plt
import seaborn as sns
from data.mnist.mnist_multinormal import pretrain_mnistmultinormal_model,load_mnist_multinormal_train
from data.mnist.mnist_0123 import pretrain_mnist_model_vae

###########################################################################################
#                                   Auxiliary functions                                   #
###########################################################################################

##########################
# Prequential evaluation #
##########################

import torch

def get_model_size(model):
    """
    通用模型大小统计函数
    - 支持 Keras (tf.keras.Model)
    - 支持 PyTorch (nn.Module)
    - 若不匹配，则返回0
    """
    total_params = 0

    # ===== ① Keras 模型 =====
    if isinstance(model, tf.keras.Model):
        total_params = model.count_params()

    # ===== ② PyTorch 模型 =====
    elif hasattr(model, "parameters"):
        try:
            total_params = sum(p.numel() for p in model.parameters())
        except Exception:
            total_params = 0

    # ===== ③ Numpy 结构（备用） =====
    elif hasattr(model, "__dict__"):
        for _, val in model.__dict__.items():
            if isinstance(val, np.ndarray):
                total_params += val.size

    # ===== 计算内存 (MB) =====
    size_mb = total_params * 4 / (1024 ** 2)  # float32 假设
    return total_params, size_mb


def update_preq_metric(s_prev, n_prev, correct, fading_factor):
    s = correct + fading_factor * s_prev
    n = 1.0 + fading_factor * n_prev
    metric = s / n

    return s, n, metric

###########
# iForest #
###########


def train_iforest(q, iforest):
    # convert queue to array
    q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in q]
    arr_unlabelled = np.concatenate(q_lst, axis=0)

    # train
    iforest.fit(arr_unlabelled)

def train_gmm(q, gmm):
    # convert queue to array
    q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in q]
    arr_unlabelled = np.concatenate(q_lst, axis=0)

    # train
    gmm.fit(arr_unlabelled)

def train_lof(q, lof):
    # convert queue to array
    q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in q]
    arr_unlabelled = np.concatenate(q_lst, axis=0)

    # train
    lof.fit(arr_unlabelled)


def lookahead_predictions_iforest(t, d_env, iforest):
    lookahead_examples = d_env['data_arr'][t:t + d_env['update_time'], :-1]
    num_lookahead_examples = lookahead_examples.shape[0]

    lookahead_pred_class = iforest.predict(lookahead_examples)
    lookahead_pred_class[lookahead_pred_class == 1] = 0
    lookahead_pred_class[lookahead_pred_class == -1] = 1

    lookahead_pred_class = pd.Series(lookahead_pred_class)
    lookahead_pred_class.index = range(t, t + num_lookahead_examples)
    lookahead_pred_class = lookahead_pred_class.astype(int)

    return lookahead_pred_class

def lookahead_predictions_lof(t, d_env, lof):
    lookahead_examples = d_env['data_arr'][t:t + d_env['update_time'], :-1]
    num_lookahead_examples = lookahead_examples.shape[0]

    lookahead_pred_class = lof.predict(lookahead_examples)
    lookahead_pred_class[lookahead_pred_class == 1] = 0
    lookahead_pred_class[lookahead_pred_class == -1] = 1

    lookahead_pred_class = pd.Series(lookahead_pred_class)
    lookahead_pred_class.index = range(t, t + num_lookahead_examples)
    lookahead_pred_class = lookahead_pred_class.astype(int)

    return lookahead_pred_class

def lookahead_predictions_gmm(t, d_env, gmm):
    lookahead_examples = d_env['data_arr'][t:t + d_env['update_time'], :-1]
    num_lookahead_examples = lookahead_examples.shape[0]
    density = gmm.score_samples(lookahead_examples)
    lookahead_pred_class = density
    percentile = 80
    threshold = np.percentile(density, 100 - percentile)
    lookahead_pred_class[lookahead_pred_class < threshold] = 1
    lookahead_pred_class[lookahead_pred_class >= threshold] = 0
    print(lookahead_pred_class)
    lookahead_pred_class = pd.Series(lookahead_pred_class)
    lookahead_pred_class.index = range(t, t + num_lookahead_examples)
    lookahead_pred_class = lookahead_pred_class.astype(int)

    return lookahead_pred_class


###############
# Autoencoder #
###############

def create_new_autoencoder(d_env):
    ae = None
    if d_env['method'] == 'ae':
        ae = d_env['fun_create_ae'](d_env)
    elif d_env['method'] == 'vae':
        ae = d_env['fun_create_vae'](d_env)


    return ae


def calc_vae_loss_vec(data, reconstruction, loss_function, z_log_var, z_mean, beta):
    ae_loss = calc_reconstruction_loss_vec(data, reconstruction, loss_function)
    kl_loss = calc_kl_loss_vec(z_log_var, z_mean, beta)  # NOTE: includes beta factor
    ae_loss = tf.cast(ae_loss, dtype=tf.float64)
    kl_loss = tf.cast(kl_loss, dtype=tf.float64)
    total_loss = ae_loss + kl_loss

    return total_loss



def loss_function_sta(d_env,ae,t):
    example = d_env['data_arr'][t:t+1, :-1]
    total_loss_vec = None

    if d_env['method'] == 'ae':

        decoded_arr = ae.prediction(example)
        total_loss_vec = calc_reconstruction_loss_vec(example, decoded_arr, ae.loss_function_name)


    elif d_env['method'] == 'vae':
        z_mean, z_log_var, decoded_arr = ae.prediction(example)
        total_loss_vec = calc_vae_loss_vec(example, decoded_arr, ae.loss_function_name, z_log_var, z_mean,
                                           ae.beta)

    return total_loss_vec





def train_autoencoder(q, d_env, ae):
    # convert queue to array
    q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in q]
    arr_unlabelled = np.concatenate(q_lst, axis=0)
    # train
    ae.train(data=arr_unlabelled, verbose=0)

    # predict and calculate loss - TODO: avoid repetition (see below)
    total_loss_vec = None
    if d_env['method'] == 'ae':
        decoded_arr = ae.prediction(arr_unlabelled)
        total_loss_vec = calc_reconstruction_loss_vec(arr_unlabelled, decoded_arr, ae.loss_function_name)


    elif d_env['method'] == 'vae':
        z_mean, z_log_var, decoded_arr = ae.prediction(arr_unlabelled)
        total_loss_vec = calc_vae_loss_vec(arr_unlabelled, decoded_arr, ae.loss_function_name, z_log_var, z_mean,
                                           ae.beta)
    if d_env['adaptive'] is 'no':
        per =d_env['ae_threshold_percentile']
    else:
    ####adaptive threshold
        q_loss = total_loss_vec
       #get mean and std of the queue
        q_mean = np.mean(q_loss)
        q_std = np.std(q_loss)

        # sort the array in ascending order
        arr_sorted = np.sort(q_loss)

        # find the index where a value of 'q_mean+q_std' would be inserted to maintain order
        idx = np.searchsorted(arr_sorted, q_mean + q_std)
        if d_env['index']=='2std':
            idx = np.searchsorted(arr_sorted, q_mean + 2*q_std)


        # calculate the percentile by dividing the index by the length of the array
        per = idx / len(arr_sorted)
        per = per * 100

    print('new percentile', per)
    threshold = np.percentile(total_loss_vec, per)
    print(threshold, 'threshold')
    return threshold






def lookahead_predictions_autoencoder(t, d_env, ae, ae_threshold):

    print(t, '-', t + d_env['update_time'], ' predict')
    lookahead_examples = d_env['data_arr'][t:t + d_env['update_time'], :-1]

    num_lookahead_examples = lookahead_examples.shape[0]

    # predict and calculate loss - TODO: avoid repetition (see above)
    total_loss_vec = None
    if d_env['method'] == 'ae':
        decoded_arr = ae.prediction(lookahead_examples)
        total_loss_vec = calc_reconstruction_loss_vec(lookahead_examples, decoded_arr, ae.loss_function_name)


    elif d_env['method'] == 'vae':
        z_mean, z_log_var, decoded_arr = ae.prediction(lookahead_examples)
        total_loss_vec = calc_vae_loss_vec(lookahead_examples, decoded_arr, ae.loss_function_name, z_log_var, z_mean,
                                           ae.beta)

    lookahead_pred_class = pd.Series(total_loss_vec > ae_threshold)
    lookahead_pred_class.index = range(t, t + num_lookahead_examples)
    lookahead_pred_class = lookahead_pred_class.astype(int)

    return lookahead_pred_class



def window_prediction_percentile(t,d_env, ae,ae_threshold):
    # predict and calculate loss
    loss_value= loss_function_sta(d_env, ae, t)

    prediction_class = pd.Series(loss_value > ae_threshold)
    prediction_class.index = range(t, t + 1)
    prediction_class = prediction_class.astype(int)
    prediction_class=prediction_class.item()
    return prediction_class

def window_prediction_ktest(t,d_env, ae,ref_loss):

    ##t test on loss value of current t with mean ref loss value
    example_loss= loss_function_sta(d_env,ae,t)
    result = stats.ks_2samp(example_loss, ref_loss)
    splitByComma = str(result).split(',')
    pValue = float(splitByComma[1].replace('pvalue=', '').strip()[:-1])

    prediction_class = pd.Series(pValue < 0.05)
    prediction_class.index = range(t, t + 1)
    prediction_class = prediction_class.astype(int)
    return prediction_class


def window_prediction_chebyshevs(t,d_env, ae,ref_win_arr,loss_arr_ref):
    # loss of current example
    example_loss= loss_function_sta(d_env,ae,t)

    #get mean and deviation of reference window.
    mu=np.mean(loss_arr_ref)
    sigma=np.std(loss_arr_ref)
    #determine a k, check if it is normal
    k=3
    c=0
    if abs(example_loss - mu) > k * sigma:
        c=c+1
    prediction_class = pd.Series(c>1/np.sqrt(k))
    prediction_class.index = range(t, t + 1)
    prediction_class = prediction_class.astype(int)
    return prediction_class










###########################################################################################
#                                           Run                                           #
###########################################################################################


def run(params_env):
    esnum = params_env['Esnum']
    retrain_triggered=False

    ######################
    # Init preq. metrics #
    ######################


    # class accuracies
    keys = range(params_env['num_classes'])
    preq_class_accs = {k: [] for k in keys}
    preq_class_acc = dict(zip(keys, [1.0, ] * params_env['num_classes']))  # NOTE: init to 1.0 not 0.0
    preq_class_acc_s = dict(zip(keys, [0.0, ] * params_env['num_classes']))
    preq_class_acc_n = dict(zip(keys, [0.0, ] * params_env['num_classes']))


    # gmean
    preq_gmeans = []

    ###initialization for recall and specificity
    preq_recalls = np.zeros(params_env['time_steps'])
    preq_specificities = np.zeros(params_env['time_steps'])



    preq_recall, preq_specificity = (1.0,) * 2  # NOTE: init to 1.0 not 0.0
    preq_recall_s, preq_recall_n = (0.0,) * 2
    preq_specificity_s, preq_specificity_n = (0.0,) * 2


    #real y
    real_y=[]





    #################################################
    # Init unsupervised learning (AE, VAE, iForest) #
    #################################################

    # init so that the compiler doesn't give warnings
    ae = None
    iforest = None
    gmm = None
    lof=None
    lookahead_pred_class = None

    # sliding window
    q_unlabelled_first = deque(params_env['data_init_unlabelled_first'], maxlen=2000)


    mov_win= deque(maxlen=params_env['unsupervised_win_size'])
    mov_warn = deque(maxlen=params_env['unsupervised_win_size'])
    ref_win = deque(maxlen=params_env['unsupervised_win_size'])
    mov_train_size=params_env['mov_win_size']
    # mov_train_size = 1000
    mov_train = deque(params_env['data_init_unlabelled'],maxlen=mov_train_size)
    #Initiate warning and alarm flag
    flag_warn=False
    flag_alarm=False
    flag_warn_count=0
    warn_moment=0

    ###ensemble_vae_DD

    lookahead_pred_class_ensemble = []

    vae_threshold_ensemble = deque(maxlen=esnum)
    vae_ensemble = deque(maxlen=esnum)
    pred_class_ensemble = deque(maxlen=esnum)
    ref_win_vae_ensemble = deque(maxlen=esnum)
    mov_win_vae_ensemble= deque(maxlen=esnum)
    ref_win_loss_vae_ensemble= deque(maxlen=esnum)
    #



    for _ in range(esnum):
        inner_deque = deque(maxlen=params_env['unsupervised_win_size'])
        ref_win_vae_ensemble.append(inner_deque)
    for _ in range(esnum):
        inner_deque_2 = deque(maxlen=params_env['unsupervised_win_size'])
        mov_win_vae_ensemble.append(inner_deque_2)



    flag_warn_vae_ensemble = deque([False] * esnum)
    warn_moment_vae_ensmeble = deque([False] * esnum)
    flag_alarm_vae_ensemble = deque([False] * esnum)
    flagalarm_count_vae_ensemble = deque([0] * esnum)



    flagalarm=False
    flagalarm_count=0

    ensemble_train=deque(maxlen=mov_train_size)
    ensemble_train_warn=deque(maxlen=params_env['unsupervised_win_size'])
    ensemble_train_trig=0
    ensemble_train_trig_warn = 0

    # box_array=[]
    mov_train_ae = deque(maxlen=mov_train_size)
    # for ensemble_window
    mov_train_vae = deque(maxlen=mov_train_size)
    p_update = deque([0] * esnum)
    mov_train_vae_ensem = deque(maxlen=10)
    random.seed(7654)

    for i in range(esnum):
        if 'mnist' in params_env['data_source']:
            p_update[i] = random.randint(mov_train_size - 500, mov_train_size + 500)
        else:
            p_update[i] = random.randint(mov_train_size - 1000, mov_train_size + 1000)

        inner_deque = deque(maxlen=p_update[i])
        mov_train_vae_ensem.append(inner_deque)
        print('win size', i, p_update[i])

    # create classifier, train & lookahead predict
    if params_env['method'] in ['ae', 'vae']:
        #create and init autoencoder
        ae = create_new_autoencoder(params_env)
        q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in q_unlabelled_first]
        arr_unlabelled = np.concatenate(q_lst, axis=0)
        # pretrain
        ae.train(data=arr_unlabelled, verbose=0)




        # predict and calculate loss - TODO: avoid repetition (see below)
        total_loss_vec = None
        if params_env['method'] == 'ae':
            decoded_arr = ae.prediction(arr_unlabelled)
            total_loss_vec = calc_reconstruction_loss_vec(arr_unlabelled, decoded_arr,
                                                              ae.loss_function_name)

        elif params_env['method'] == 'vae':
            z_mean, z_log_var, decoded_arr = ae.prediction(arr_unlabelled)
            total_loss_vec = calc_vae_loss_vec(arr_unlabelled, decoded_arr, ae.loss_function_name, z_log_var,
                                               z_mean,
                                               ae.beta)
        ###same as train_autoencoder, to get the initial percentile value, used for incremental learning.

        if params_env['adaptive'] is 'no':
            per = params_env['ae_threshold_percentile']
        else:
            ####adaptive threshold
            q_loss = total_loss_vec
            print('retrain time for percentile')
            print(q_loss, 'q_loss')
            q_mean = np.mean(q_loss)
            q_std = np.std(q_loss)

            # sort the array in ascending order
            arr_sorted = np.sort(q_loss)

            # find the index where a value of 'q_mean+q_std' would be inserted to maintain order
            idx = np.searchsorted(arr_sorted, q_mean + q_std)
            if params_env['index'] == '2std':
                idx = np.searchsorted(arr_sorted, q_mean + 2 * q_std)

            # calculate the percentile by dividing the index by the length of the array
            per = idx / len(arr_sorted)
            per = per * 100

        if params_env['strategy'] == 'ensemble_ae'or params_env['strategy'] == 'hybrid'or params_env['strategy'] == 'ensemble_vae_DD'or params_env['strategy'] == 'ensemble_incr'or params_env['strategy'] == 'ensemble_base' or params_env['strategy'] == 'ensemble_vae_new':
            params_env['update_time'] = int(mov_train_size * params_env['unsupervised_win_size_update'])

        ae_threshold = np.percentile(total_loss_vec, per)
        lookahead_pred_class = lookahead_predictions_autoencoder(0, params_env, ae, ae_threshold)
        print(lookahead_pred_class)


    if params_env['strategy']== 'ensemble_vae_DD':
        # create 10 VAE
        params_env['method'] = 'vae'
        i=0
        params_env_copy = params_env.copy()
        for b in range(esnum):
            params_env_copy = params_env.copy()
            params_env_copy['beta'] = params_env['beta']
            vae = create_new_autoencoder(params_env_copy)
            encoder_weights = vae.encoder.get_weights()
            # print(encoder_weights,i,'vae initial weights')
            i = i + 1
            vae_threshold = train_autoencoder(q_unlabelled_first, params_env_copy, vae)
            lookahead_pred_class = lookahead_predictions_autoencoder(0, params_env_copy, vae, vae_threshold)
            vae_ensemble.append(vae)
            lookahead_pred_class_ensemble.append(lookahead_pred_class)
            vae_threshold_ensemble.append(vae_threshold)



    elif params_env['method'] == 'iforest':
        print('run with iforest')
        iforest = params_env['fun_create_iforest'](params_env)
        train_iforest(q_unlabelled_first, iforest)
        lookahead_pred_class = lookahead_predictions_iforest(0, params_env, iforest)

    elif params_env['method'] == 'lof':
        print('run with lof')
        lof = params_env['fun_create_lof'](params_env)
        train_lof(q_unlabelled_first, lof)
        lookahead_pred_class = lookahead_predictions_gmm(0, params_env,lof)

    elif params_env['method'] == 'gmm':
        print('run with gmm')
        gmm = params_env['fun_create_gmm'](params_env)
        train_gmm(q_unlabelled_first, gmm)
        lookahead_pred_class = lookahead_predictions_gmm(0, params_env,gmm)



    #########
    # Start #
    #########
    retrain_time = []
    warn_time = []
    alarm_time=[]

    i = 0

    q_unlabelled= deque(params_env['data_init_unlabelled'], maxlen=params_env['unsupervised_win_size'])

    correct_num=0
    false_pos=0
    true_pos=0
    false_neg=0
    true_neg=0
    false_pos_b=0
    true_pos_b=0
    false_neg_b=0
    true_neg_b=0
    false_pos_a = 0
    true_pos_a = 0
    false_neg_a = 0
    true_neg_a = 0



    ####store prediction and anomaly score
    pred_list = []
    score_list=[]
    print(len(params_env['data_arr']),'data length')

    for t in range(0, params_env['time_steps']):
        # start = time.time()
        if t % 500 == 0:
            print('Time step: ', t)


        #################
        # Concept drift #
        #################

        # reset preq. metrics
        if 'drift' in params_env['data_source']:
            flag_reset_metric = False

            if isinstance(params_env['t_drift'], int) and t == params_env['t_drift']:
                flag_reset_metric = True
            elif isinstance(params_env['t_drift'], tuple) and \
                    (t == params_env['t_drift'][0] or t == params_env['t_drift'][1]):
                flag_reset_metric = True

            if flag_reset_metric:
                # preq_general_acc_n = 0.0
                # preq_general_acc_s = 0.0

                preq_class_acc = dict(zip(keys, [1.0, ] * params_env['num_classes']))  # NOTE: init to 1.0 not 0.0
                preq_class_acc_s = dict(zip(keys, [0.0, ] * params_env['num_classes']))
                preq_class_acc_n = dict(zip(keys, [0.0, ] * params_env['num_classes']))

         ###############
        # Get example #
        ###############


        xy = params_env['data_arr'][t, :]

        x = xy[:-1]  # of shape (n,)
        x = np.reshape(x, (1, x.shape[0]))
        y = xy[-1]

####get anomaly score
        ####baseline, hybid
        if params_env['method']=='vae' and  params_env['strategy']!='ensemble_vae_DD':
            z_mean, z_log_var, decoded_x = ae.prediction(x)
            score_t = calc_vae_loss_vec(x, decoded_x, ae.loss_function_name, z_log_var,z_mean,ae.beta)
        elif params_env['strategy']=='ensemble_vae_DD':
            vae=vae_ensemble[0]
            z_mean, z_log_var, decoded_x = vae.prediction(x)
            score_t = calc_vae_loss_vec(x, decoded_x, vae.loss_function_name, z_log_var,z_mean,vae.beta)
        elif params_env['method'] in ['iforest']:
            score_t =-iforest.score_samples(x)
        elif params_env['method'] in ['lof']:
            score_t =-lof.score_samples(x)

        # ==== 确保 score_t 是 float 数字 ====
        if isinstance(score_t, (np.ndarray, list)):
            # 若是数组或列表，取第一个元素
            score_t = float(np.array(score_t).reshape(-1)[0])
        else:
            score_t = float(score_t)


        score_list.append(score_t)
        # print('score at:',t,':',score_t)


        #######################
        # AE: Predict & Train #
        #######################

        # predict current example for strategy baseline
        if params_env['strategy']=='baseline':
            pred_class = lookahead_pred_class[t]
        if  params_env['strategy']=='hybrid':
            if t < params_env['update_time']:
                pred_class = lookahead_pred_class[t]

        if params_env['strategy'] == 'ensemble_vae_DD' :

            if t < params_env['update_time']:

                sum_pred_class = sum(sublist[t] for sublist in lookahead_pred_class_ensemble)
                if sum_pred_class >=params_env['Pthre']:
                    pred_class = 1
                else:
                    pred_class = 0


        # append example to sliding window (baseline)
        if params_env['strategy']=='baseline':
            q_unlabelled.append(x)

        # training and lookahead predictions for strategy 'baseline'
        if params_env['strategy'] == 'baseline' and t != 0 and t != params_env['time_steps'] - 1 and (t + 1) % params_env['update_time'] == 0:

            print(t, ' train')
            retrain_time.append(t)
            if params_env['method'] in ['ae', 'vae']:
                ae_threshold = train_autoencoder(q_unlabelled, params_env, ae)
                lookahead_pred_class= lookahead_predictions_autoencoder(t + 1, params_env, ae, ae_threshold)
                # print(len(q_unlabelled))
                print(ae_threshold, t)


            elif params_env['method'] == 'iforest':
                print('train with iforest')
                if not params_env['unsupervised_flag_incremental']:
                    iforest = params_env['fun_create_iforest'](params_env)
                ####incremental learning
                train_iforest(q_unlabelled, iforest)
                lookahead_pred_class = lookahead_predictions_iforest(t + 1, params_env, iforest)
            elif params_env['method'] == 'lof':
                print('train with lof')
                ####incremental learning
                train_lof(q_unlabelled, lof)
                lookahead_pred_class = lookahead_predictions_lof(t + 1, params_env, lof)
            elif params_env['method'] == 'gmm':
                print('train with gmm')
                train_gmm(q_unlabelled, gmm)
                lookahead_pred_class = lookahead_predictions_gmm(t + 1, params_env, gmm)



        elif params_env['strategy'] == 'hybrid':
            if params_env['method'] in ['ae', 'vae']:
                # prediction for current t (based on percentile)
                if not t < params_env['update_time']:
                    pred_class = window_prediction_percentile(t, params_env, ae, ae_threshold)

            mov_train.append(x)
            if len(ref_win) == params_env['unsupervised_win_size']:
                q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in ref_win]
                ref_win_arr = np.concatenate(q_lst, axis=0)

                if params_env['method'] == 'ae':
                    decoded_arr = ae.prediction(ref_win_arr)
                    ref_win_loss  = calc_reconstruction_loss_vec(ref_win_arr, decoded_arr,
                                                                      ae.loss_function_name)

                if params_env['method'] == 'vae':
                    z_mean, z_log_var, decoded_arr = ae.prediction(ref_win_arr)
                    ref_win_loss = calc_vae_loss_vec(ref_win_arr, decoded_arr, ae.loss_function_name, z_log_var,
                                                     z_mean,
                                                     ae.beta)
                mov_win.append(x)
            else:
                ref_win.append(x)



            if t != 0 and t != params_env['time_steps']- 1 and (t + 1) % params_env['update_time']  == 0 and flag_warn==False:
                print(t, 'incre train')
                print('length of mov_train at incr. training',len(mov_train))
                retrain_time.append(t)
                start_incr=time.time()
                if params_env['method'] in ['ae', 'vae']:
                    ae_threshold = train_autoencoder(mov_train, params_env, ae)
                end_incr = time.time()
                time_incr=end_incr-start_incr
                print('incremental learning time:',time_incr)
            # if the moving window is full, then check for the concept drift detection and retrain if necessary.
            if len(mov_win) == params_env['unsupervised_win_size']:
                ###conduct drift detection
                # convert queue to array: moving window
                q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in mov_win]
                mov_win_arr = np.concatenate(q_lst, axis=0)
             # drift detection
                total_loss_vec = None
                if params_env['method'] == 'ae':
                    decoded_arr = ae.prediction(ref_win_arr)
                    ref_win_loss = calc_reconstruction_loss_vec(ref_win_arr, decoded_arr,
                                                                ae.loss_function_name)
                    decoded_arr = ae.prediction(mov_win_arr)
                    mov_win_loss = calc_reconstruction_loss_vec(mov_win_arr, decoded_arr, ae.loss_function_name)

                if params_env['method'] == 'vae':
                    z_mean, z_log_var, decoded_arr = ae.prediction(ref_win_arr)
                    ref_win_loss = calc_vae_loss_vec(ref_win_arr, decoded_arr, ae.loss_function_name, z_log_var,
                                                     z_mean,
                                                     ae.beta)
                    z_mean, z_log_var, decoded_arr = ae.prediction(mov_win_arr)
                    mov_win_loss = calc_vae_loss_vec(mov_win_arr, decoded_arr, ae.loss_function_name, z_log_var,
                                                     z_mean,
                                                     ae.beta)


                result = mannwhitneyu(mov_win_loss, ref_win_loss)

                splitByComma = str(result).split(',')
                pValue = float(splitByComma[1].replace('pvalue=', '').strip()[:-1])
                if t in [199, 200,500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,5500,6000,6500,7000,8000,9000,10000,15000,20000]:
                    print(t)
                    print(result)
                    print(pValue)


                if flag_warn == False:
                    ###

                    #
                    if pValue < 0.01:
                        flag_warn = True
                        #print(pValue)
                        print('warn true', t,pValue)
                        warn_moment=t
                        warn_time.append(t)
                        flag_warn_count += 1


                if pValue < params_env['palarm']:

                    flag_alarm = True
                    alarm_time.append(t)

                    print(pValue)
                    print('alarm true', t)


            if flag_warn == True and flag_alarm==False:
                mov_warn.append(x)
                #if flag raised for more than expiry time, empty mov_warn
                if t > warn_moment+200:
                    flag_warn =False
                    mov_warn = deque(maxlen=params_env['unsupervised_win_size'])
                    flag_warn_count = 0


            if flag_alarm == True:


                #create new AE
                if params_env['method'] in ['ae', 'vae']:
                    start_retrain=time.time()
                    ae = create_new_autoencoder(params_env)
                    ae_threshold = train_autoencoder(mov_warn, params_env, ae)
                    end_retrain = time.time()
                    time_retrain=end_retrain-start_retrain
                    print('retrain time after drift:',time_retrain)

                print(len(mov_warn))

                #empty all windows
                mov_train = deque(maxlen=mov_train_size)
                mov_win = deque(maxlen=params_env['unsupervised_win_size'])
                ref_win = deque(maxlen=params_env['unsupervised_win_size'])
                mov_warn = deque(maxlen=params_env['unsupervised_win_size'])
                flag_alarm = False
                flag_warn = False
                flag_warn_count = 0

                print('detection retrained at t:', t)
                retrain_time.append(t)



        elif params_env['strategy'] == 'ensemble_vae_DD':
            if not t < params_env['update_time']:
                i = 0
                pred_class_ensemble = []

                # ========== ensemble voting prediction ==========
                for b in range(esnum):
                    params_env_copy = params_env.copy()
                    params_env_copy['beta'] = params_env['beta']
                    vae = vae_ensemble[i]

                    vae_threshold = vae_threshold_ensemble[i]
                    pred_class_vae = window_prediction_percentile(t, params_env_copy, vae, vae_threshold)
                    pred_class_ensemble.append(pred_class_vae)
                    i += 1

                # === voting strategy ===
                if sum(pred_class_ensemble) >= params_env['Pthre']:
                    pred_class = 1
                else:
                    pred_class = 0

                # === append current data for ensemble moving windows ===
                for i in range(esnum):
                    mov_train_vae_ensem[i].append(x)

                # === trigger reset for normal ensemble training ===
                if ensemble_train_trig > 0 and t > ensemble_train_trig + mov_train_size:
                    ensemble_train_trig = 0

                # === new improved append logic for warning collection ===
                if ensemble_train_trig_warn > 0 and ensemble_train_trig == 0:
                    ensemble_train_warn.append(x)
                    if len(ensemble_train_warn) == 1:
                        print(f"[APPEND] start collecting warning samples at t={t}")
                    elif len(ensemble_train_warn) % 10 == 0:
                        print(f"[APPEND] collected {len(ensemble_train_warn)} warning samples")
                    # === clear expired warning collection ===
                    if t > ensemble_train_trig_warn + 100:
                        ensemble_train_warn = deque(maxlen=params_env['unsupervised_win_size'])
                        ensemble_train_trig_warn = 0

                # ======================================================
                # incremental learning (periodic update by different window sizes)
                # ======================================================

                for i in range(esnum):
                    if t != 0 and t != params_env['time_steps'] - 1 and (t + 1) % p_update[i] == 0:
                        retrain_time.append(t)
                        print(i, p_update[i])
                        if not flag_warn_vae_ensemble[i]:
                            params_env_copy = params_env.copy()
                            vae = vae_ensemble[i]
                            if params_env['incr'] == 'yes':
                                start_incr_es = time.time()
                                vae_threshold = train_autoencoder(mov_train_vae_ensem[i], params_env_copy, vae)
                                vae_ensemble[i] = vae
                                vae_threshold_ensemble[i] = vae_threshold
                                end_incr_es = time.time()
                                time_incr_es=end_incr_es-start_incr_es
                                print('time:',t,'member:',i,'ensemble incremental learning time:', time_incr_es)
                # ======================================================
                # drift detection (Dthre != 'noDD')
                # ======================================================
                if isinstance(params_env['Dthre'], int):

                    # === choose ensemble vs single detector ===
                    if params_env.get('index', 'esdd') == 'esdd':
                        detector_range = range(esnum)
                    elif params_env.get('index') == 'onedd':
                        detector_range = [0]
                    else:
                        raise ValueError(f"Unknown index type: {params_env.get('index')}")

                    retrain_triggered = False

                    # === main detection loop ===
                    for i in detector_range:
                        vae = vae_ensemble[i]

                        # --- update reference window ---
                        if len(ref_win_vae_ensemble[i]) == params_env['unsupervised_win_size']:
                            q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in
                                     ref_win_vae_ensemble[i]]
                            ref_win_arr_vae = np.concatenate(q_lst, axis=0)
                            z_mean, z_log_var, decoded_arr_vae = vae.prediction(ref_win_arr_vae)
                            ref_win_loss_vae = calc_vae_loss_vec(
                                ref_win_arr_vae, decoded_arr_vae, vae.loss_function_name, z_log_var, z_mean, vae.beta
                            )
                            ref_win_loss_vae_ensemble.append(ref_win_loss_vae)
                            mov_win_vae_ensemble[i].append(x)
                        else:
                            ref_win_vae_ensemble[i].append(x)

                        # --- drift detection ---
                        if len(mov_win_vae_ensemble[i]) == params_env['unsupervised_win_size']:
                            q_lst = [np.reshape(a, (1, a.shape[0])) if a.ndim == 1 else a for a in
                                     mov_win_vae_ensemble[i]]
                            mov_win_arr_vae = np.concatenate(q_lst, axis=0)
                            z_mean, z_log_var, decoded_arr_vae = vae.prediction(mov_win_arr_vae)
                            mov_win_loss_vae = calc_vae_loss_vec(
                                mov_win_arr_vae, decoded_arr_vae, vae.loss_function_name, z_log_var, z_mean, vae.beta
                            )

                            result_vae = mannwhitneyu(mov_win_loss_vae, ref_win_loss_vae)
                            pValue = float(str(result_vae).split(',')[1].replace('pvalue=', '').strip()[:-1])

                            # === warning ===
                            if not flag_warn_vae_ensemble[i] and pValue < 0.01:

                                flag_warn_vae_ensemble[i] = True
                                warn_moment_vae_ensmeble[i] = t
                                if ensemble_train_trig_warn == 0:
                                    ensemble_train_trig_warn = t
                                print(f"[WARN] Detector {i} p={pValue:.5f} at t={t}")

                            # === alarm ===
                            if pValue < params_env['palarm']:
                                flag_alarm_vae_ensemble[i] = True
                                flagalarm_count_vae_ensemble[i] += 1
                                if flagalarm_count_vae_ensemble[i] == 1:
                                    flagalarm_count += 1
                                if ensemble_train_trig == 0:
                                    ensemble_train_trig = t
                                print(f"[ALARM] Detector {i} p={pValue:.5f} at t={t}")

                    # ======================================================
                    # retrain trigger logic（保持原逻辑 + 冷却期判断）
                    # ======================================================
                    if params_env.get('index', 'esdd') == 'esdd':
                        if flagalarm_count >= params_env['Dthre'] :
                            retrain_triggered = True
                            print(f"🔁 retrain by ENSEMBLE voting at t={t}")
                    else:
                        if flag_alarm_vae_ensemble[0]:
                            retrain_triggered = True
                            print(f"🔁 retrain by ONE detector (i=0) at t={t}")



                    # ======================================================
                    # retrain operation (shared)
                    # ======================================================
                    if retrain_triggered:
                        n_warn = len(ensemble_train_warn)
                        print(f"[CHECK] retrain triggered at t={t}, collected {n_warn} samples")

                        if n_warn < 5:
                            print(f"[DELAY] retrain postponed: only {n_warn} warning samples")
                            retrain_triggered = False

                        if retrain_triggered:
                            print(f"[RETRAIN] start retrain at t={t} with {len(ensemble_train_warn)} samples")
                            warn_time.append(ensemble_train_trig_warn)
                            alarm_time.append(t)
                            start_retrain_es = time.time()
                            for b in range(esnum):
                                params_env_copy = params_env.copy()
                                params_env_copy['beta'] = params_env['beta']
                                vae = create_new_autoencoder(params_env_copy)
                                vae_threshold = train_autoencoder(ensemble_train_warn, params_env_copy, vae)
                                vae_ensemble[b] = vae
                                vae_threshold_ensemble[b] = vae_threshold
                            end_retrain_es = time.time()
                            time_retrain_es=end_retrain_es-start_retrain_es
                            print('ensem retrain time after drift:', time_retrain_es)
                            import sys
                            sys.exit()

                            for i in range(esnum):
                                inner_deque = deque(maxlen=p_update[i])
                                mov_train_vae_ensem.append(inner_deque)

                            # --- reset all states ---
                            mov_win_vae_ensemble = deque(maxlen=esnum)
                            ref_win_vae_ensemble = deque(maxlen=esnum)
                            for _ in range(esnum):
                                inner_deque = deque(maxlen=params_env['unsupervised_win_size'])
                                ref_win_vae_ensemble.append(inner_deque)
                            for _ in range(esnum):
                                inner_deque_2 = deque(maxlen=params_env['unsupervised_win_size'])
                                mov_win_vae_ensemble.append(inner_deque_2)

                            # ensemble_train=deque(maxlen=mov_train_size)
                            ensemble_train_trig=0
                            ensemble_train_warn = deque(maxlen=params_env['unsupervised_win_size'])
                            ensemble_train_trig_warn = 0
                            flagalarm_count = 0
                            flagalarm_count_vae_ensemble = deque([0] * esnum)
                            flag_alarm_vae_ensemble = deque([False] * esnum)
                            flag_warn_vae_ensemble = deque([False] * esnum)
                            retrain_triggered = False
                            print("[RESET] retrain finished and state cleared\n")

                            # =========================
                            # [NEW] 设置下一次可重训的最早时间（冷却开始）
                            # # =========================
                            # params_env['next_retrain_earliest_t'] = t + params_env['retrain_gap']
                            # print(
                            #     f"[COOLDOWN] retrain cooldown set: next earliest t = {params_env['next_retrain_earliest_t']}")
                            # # =========================
        # end = time.time()
        # print('time per step:' ,end-start)
        ###############
        # Correctness #
        ###############
        correct = 1 if y == pred_class else 0  # check if prediction was correct
        correct_num+=correct
        if y==0 and pred_class==1:
            false_pos+=1
        if y==0 and pred_class==0:
            true_neg+=1
        if y == 1 and pred_class == 0:
            false_neg+=1
        if y == 1 and pred_class == 1:
            true_pos+=1


        # if len(alarm_time)==0:
        if isinstance(params_env['t_drift'], int) or (isinstance(params_env['t_drift'], tuple) and t < params_env['t_drift'][0]):
            if y == 0 and pred_class == 1:
                false_pos_b += 1
            if y == 0 and pred_class == 0:
                true_neg_b += 1
            if y == 1 and pred_class == 0:
                false_neg_b += 1
            if y == 1 and pred_class == 1:
                true_pos_b += 1

        else:
            if y == 0 and pred_class == 1:
                false_pos_a += 1
            if y == 0 and pred_class == 0:
                true_neg_a += 1
            if y == 1 and pred_class == 0:
                false_neg_a += 1
            if y == 1 and pred_class == 1:
                true_pos_a += 1



        real_y.append(y)
        pred_list.append(pred_class)

        if y==0:
            preq_specificity_s, preq_specificity_n, preq_specificity = update_preq_metric(preq_specificity_s,
                                                                                          preq_specificity_n, correct,
                                                                                          params_env['preq_fading_factor'])
        else:
            preq_recall_s, preq_recall_n, preq_recall = update_preq_metric(preq_recall_s, preq_recall_n, correct,
                                                                           params_env['preq_fading_factor'])

        preq_recalls[t] = preq_recall
        preq_specificities[t] = preq_specificity
        ########################
        # Update preq. metrics #
        ########################



        preq_class_acc_s[y], preq_class_acc_n[y], preq_class_acc[y] = update_preq_metric(
            preq_class_acc_s[y], preq_class_acc_n[y], correct, params_env['preq_fading_factor'])

        lst = []
        for k, v in preq_class_acc.items():
            preq_class_accs[k].append(v)
            lst.append(v)


        gmean = np.power(np.prod(lst), 1.0 / len(lst))

        preq_gmeans.append(gmean)







    print(correct_num,'correst_num')
    print(false_pos,'false pos')
    print(true_pos, 'true pos')
    print(false_neg, 'false neg')
    print(true_neg, 'true neg')
    print(false_pos_b,'false pos b')
    print(true_pos_b, 'true pos b')
    print(false_neg_b, 'false neg b')
    print(true_neg_b, 'true neg b')
    print(false_pos_a,'false pos a')
    print(true_pos_a, 'true pos a')
    print(false_neg_a, 'false neg a')
    print(true_neg_a, 'true neg a')
    # Open a file in write mode
    out_dir = 'exps/'
    file_path=os.path.join(os.getcwd(), out_dir, str(params_env['data_source']) + str(params_env['strategy'])+str(params_env['Pthre'])+str(params_env['Dthre']) +str(params_env['Esnum']) +'_metric.txt')
    file = open(file_path, "w")
    # Write values to the file
    values = [false_pos,true_pos,false_neg,true_neg,false_pos_b,true_pos_b,false_neg_b,true_neg_b,false_pos_a,true_pos_a,false_neg_a,true_neg_a]
    for value in values:
        file.write(str(value) + "\n")
    # Close the file
    file.close()

    return preq_class_accs, preq_gmeans, preq_recalls,preq_specificities,retrain_time,warn_time,alarm_time,pred_list,score_list



