# -*- coding: utf-8 -*-
import os
import pandas as pd
# GPU-related code

# hide warnings (before importing Keras)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# choose GPU (before importing Keras)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# dynamically grow memory
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.3
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

import numpy as np
from main_exp import run
# from class_nn_ae_vanilla import AE
from class_nn_ae import AE
from class_nn_ae_variational import VAE
from data.mnist.mnist import load_mnist
from data.fraud.fraud import load_fraud,load_fraud_pre,load_fraud_init,load_fraud_new,load_fraud_recurrent

from data.forest.forest import load_forest,load_forest_init,load_forest_pre,load_forest_new
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from data.mnist.mnist_0123 import load_mnist_init,load_mnist_arriving

from data.mnist.mnist_multinormal import load_mnist_multinormal_init,load_mnist_multinormal_arriving
from data.data_generate_sea import load_sea_new,load_sea_init
from data.data_generate_circle import load_circle,load_circle_init,load_circle_new,load_vib,load_vib_init
from data.data_generate_sine import load_sine,load_sine_init
from data.arrhythmia.arrhy import load_arrhy,load_arrhy_init
from sklearn.neighbors import LocalOutlierFactor

#from confusion_matrix import newrun
###########################################################################################
#                                   Auxiliary functions                                   #
###########################################################################################

###For GPU server
# Selecting the GPU always after import tensorflow
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="2" # 0 or 1 depending on the GPU you want
# Debugging print out options
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
##allocate memory (0-1,percentage of memory)

tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.compat.v1.Session(config=config)







#######
# I/O #
#######


# Create text file
def create_file(filename):
    f = open(filename, 'w')
    f.close()

def write_simple(filename, arr):
    arr = np.array(arr, dtype=np.float32)
    with open(filename, 'a') as f:
        np.savetxt(f, [arr], delimiter=', ', fmt='%1.6f')

# Write array to a row in the given file
def write_to_file(filename, arr,retrain_time,warn_time,alarm_time):
    with open(filename, 'a') as f:
        len_retrain=np.float(len(retrain_time))
        retrain_time=np.array(retrain_time, dtype=np.float32)
        len_warn = np.float(len(warn_time))
        warn_time = np.array(warn_time, dtype=np.float32)
        len_alarm= np.float(len(alarm_time))
        alarm_time = np.array(alarm_time, dtype=np.float32)

        arr=np.concatenate((arr,retrain_time.T, warn_time.T, alarm_time.T))
        arr = np.append(arr, len_retrain)
        arr = np.append(arr, len_warn)
        arr = np.append(arr, len_alarm)
        np.savetxt(f, [arr], delimiter=', ', fmt='%1.6f')

###########
# iForest #
###########


def create_iforest(d_env):
    return IsolationForest(max_samples='auto', contamination='auto', random_state=d_env['random_state'])

def create_gmm(d_env):
    return GaussianMixture(n_components=2, covariance_type='full', random_state=d_env['random_state'])

def create_lof(d_env):
    return LocalOutlierFactor(n_neighbors=3,novelty=True)

###############
# Autoencoder #
###############


def create_ae(d_env):
    layer_dims = None
    learning_rate = None
    loss_function = None


    if 'mnist' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 512, 256, 64]
        learning_rate = 0.0001
        loss_function = 'binary_crossentropy'


    elif 'sea2' in d_env['data_source']:
        layer_dims = [d_env['num_features'],64,8, 2]
        learning_rate = 0.001
        loss_function = 'binary_crossentropy'

    elif 'arrhy' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 128, 32]
        learning_rate = 0.0001
        loss_function = 'square_error'


    elif 'circle' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 8,2]
        learning_rate = 0.001
        loss_function = 'binary_crossentropy'

    elif 'vib' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 8, 2]
        learning_rate = 0.001
        loss_function = 'binary_crossentropy'
    elif 'sine' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 8, 2]
        learning_rate = 0.001
        loss_function = 'binary_crossentropy'
    elif 'fraud' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 64, 32, 8]
        learning_rate = 0.001
        loss_function = 'square_error'
    elif 'forest' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 64, 32]
        learning_rate = 0.0001
        loss_function = 'square_error'


    ae = AE(
        layer_dims=layer_dims,
        learning_rate=learning_rate,
        loss_function=loss_function,
        num_epochs=d_env['num_epochs'],
        minibatch_size=128,
        noise_gaussian_avg=d_env['noise_gaussian_avg'],
        noise_gaussian_std=d_env['noise_gaussian_std'],
        l1_activations=d_env['reg_l1'],
        reg_l2=0.0001,
        seed=d_env['seed']
    )
    ae.autoencoder.summary()

    return ae

#######
# VAE #
#######


def create_vae(d_env):
    layer_dims = None
    learning_rate = None
    loss_function = None

    if 'mnist' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 512, 256, 64]
        # learning_rate= d_env['lr']
        learning_rate = 0.0001
        loss_function = 'binary_crossentropy'
        #l1_activations=0.1,
        #dropout_rate = 0.5,
    elif 'sea2' in d_env['data_source']:
        layer_dims = [d_env['num_features'],64,8,2]
        # learning_rate = d_env['lr']
        learning_rate = 0.001
        loss_function = 'binary_crossentropy'


    elif 'arrhy' in d_env['data_source']:
        layer_dims = [d_env['num_features'],128,32]
        learning_rate = 0.0001
        loss_function = 'square_error'




    elif 'circle' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 8, 2]
        learning_rate = 0.001
        loss_function = 'binary_crossentropy'

    elif 'vib' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 8, 2]
        learning_rate = 0.001
        loss_function = 'binary_crossentropy'
    elif 'sine' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 8, 2]
        learning_rate = 0.001
        loss_function = 'binary_crossentropy'
    elif 'fraud' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 64, 32, 8]
        learning_rate = 0.001
        loss_function = 'square_error'
    elif 'forest' in d_env['data_source']:
        layer_dims = [d_env['num_features'], 64, 32]
        learning_rate = 0.0001
        loss_function = 'square_error'


    vae = VAE(
        layer_dims=layer_dims,
        learning_rate=learning_rate,
        loss_function=loss_function,
        num_epochs=d_env['num_epochs'],
        batch_size=64,
        beta=d_env['beta'],
        dropout_rate=0.0,
        seed=d_env['seed']
    )

    return vae



#################
# Safety checks #
#################


def run_safety_checks(d_env):
    # method
    if d_env['method'] not in ['ae', 'vae', 'iforest','gmm','lof']:
        raise Exception('Incorrect unsupervised learning method entered.')

    # window replacement percentage
    if d_env['unsupervised_win_size_update'] < 0.0 or d_env['unsupervised_win_size_update'] > 1.0:
        raise Exception('Percentage must be in [0,1].')



    # data source
    if d_env['data_source'] not in ['mnist_01_extreme', 'mnist_01_extreme_drift',
                                    'mnist_23_extreme', 'mnist_23_extreme_drift',
                                    'mnist_multinormal_extreme', 'mnist_multinormal_extreme_drift',
                                    'mnist_multi_extreme', 'mnist_multi_extreme_drift',
                                    'mnist_01_severe', 'mnist_01_severe_drift',
                                    'mnist_23_severe', 'mnist_23_severe_drift',
                                    'mnist_multinormal_severe', 'mnist_multinormal_severe_drift',
                                    'mnist_multi_severe', 'mnist_multi_severe_drift','fraud_severe_drift',
                                    'fraud_severe', 'fraud_extreme', 'fraud_extreme_drift',
                                    'forest_severe', 'forest_severe_drift',
                                    'forest_extreme', 'forest_extreme_drift',
                                   'sea2_severe_drift','sea2_extreme_drift','sea2_severe','sea2_extreme',
                                    'circle_severe_drift','circle_extreme_drift','circle_severe','circle_extreme',
                                    'sine_severe_drift','sine_extreme_drift','sine_severe','sine_extreme',
                                    'vib_severe_drift','vib_extreme_drift','vib_severe','vib_extreme','arrhy']:
        raise Exception('Incorrect dataset entered.')

###########################################################################################
#                                         Main                                            #
###########################################################################################


def main(params_env):
    # Input dictionary entries
    #
    # - 'repeats': number of experiment's repetitions (e.g., 20)
    # - 'data_source': dataset name
    # - 'paradigm': learning paradigm ('unsupervised' or 'active')
    # - 'method': learning method ('ae', 'vae' 'iforest' for unsupervised paradigm or 'rvus', 'actiq' for active)

    # - 'active_budget_total': active learning budget in [0, 1] (for RVUS or ActiQ)
    # - 'ae_threshold_percentile': percetile value (for AE or VAE)

    # - 'unsupervised_win_size': window size (for unsupervised paradigm)
    # - 'unsupervised_win_size_update': percentage of window replacement in [0, 1] (for unsupervised paradigm)

    ######################
    # Settings: REQUIRED #
    ######################

    # safety checks
    run_safety_checks(params_env)

    # unsupervised learning (AE, VAE, iForest)
    # params_env['num_epochs'] = 10
    params_env['unsupervised_flag_incremental'] =True
    params_env['update_time'] = int(params_env['unsupervised_win_size'] * params_env['unsupervised_win_size_update'])

    # random seed (NOTE: Keep it fixed to replicate the paper's results)
    params_env['seed'] = 7654
    params_env['random_state'] = np.random.RandomState(seed=params_env['seed'])

    #############
    # Load data #
    #############

    # load data
    load_fun = None
    if 'sea2' in params_env['data_source']:
        load_fun = load_sea_new
        print('good load_fun sea2')

    elif 'arrhy' in params_env['data_source']:
        load_fun = load_arrhy
        print('good load_fun arrhy')
    elif 'circle' in params_env['data_source']:
        load_fun = load_circle
        print('good load_fun circle')
    elif 'vib' in params_env['data_source']:
        load_fun = load_vib
    elif 'sine' in params_env['data_source']:
        load_fun = load_sine
        print('good load_fun sine')
    elif 'fraud' in params_env['data_source']:
        # load_fun = load_fraud
        load_fun = load_fraud_recurrent
    elif 'forest' in params_env['data_source']:
        load_fun = load_forest
        # load_fun = load_forest_pre

    elif '23' in params_env['data_source']:
        load_fun = load_mnist_arriving
        print('good load_fun 23')
    elif 'normal' in params_env['data_source']:
        load_fun = load_mnist_multinormal_arriving
        print('good load_fun multinormal')
    elif '01' or 'multi' in params_env['data_source']:
        load_fun = load_mnist
        print('good load_fun 01/multi')




    if 'forest' in params_env['data_source']:
        params_env['data_arr'], params_env['data_init_unlabelled'], _, params_env['t_drift'] = load_fun(data_source=params_env['data_source'], num_init_unlabelled=2000,random_state=params_env['random_state'])
        # params_env['data_arr'], params_env['data_init_unlabelled'], params_env['t_drift'] = load_fun(params_env)
        params_env['data_init_unlabelled_first'], _ = load_forest_init(n_train=2000, n_val=1000)



        # params_env['data_init_unlabelled_first'], params_env['data_arr'], params_env['data_init_unlabelled'], \
        # params_env['t_drift'] = load_forest_new()
        # print('forest_init_file_loaded')
    elif 'fraud' in params_env['data_source']:
        params_env['data_arr'], params_env['data_init_unlabelled'], _, params_env['t_drift'] = load_fun(data_source=params_env['data_source'], num_init_unlabelled=2000,random_state=params_env['random_state'])
        # params_env['data_arr'], params_env['data_init_unlabelled'], params_env['t_drift'] = load_fun(params_env)
        params_env['data_init_unlabelled_first'], _ = load_fraud_init(n_train=2000, n_val=1000)
        print(len(params_env['data_arr']), 'data length')
        # params_env['data_init_unlabelled_first'], params_env['data_arr'], params_env['data_init_unlabelled'], \
        # params_env['t_drift'] = load_fraud_new()
        # print('fraud_init_file_loaded')

    elif 'arrhy' in params_env['data_source']:
        params_env['data_arr'], params_env['data_init_unlabelled'], params_env['t_drift'] = load_fun(params_env)
        params_env['data_init_unlabelled_first'], _ = load_arrhy_init(n_train=2000, n_val=1000)
        print('good arrhy')

    elif 'sea2' in params_env['data_source']:
        params_env['data_arr'], params_env['data_init_unlabelled'], params_env['t_drift'] = load_fun(params_env)
        params_env['num_init_per_class'] = 20000
        params_env['memory_size']= 15000
        params_env['probs']=0.1
        params_env['data_init_unlabelled_first'], _  = load_sea_init(n_train=2000, n_val=1000)
        print('good sea')



    elif 'circle' in params_env['data_source']:
        params_env['data_arr'], params_env['data_init_unlabelled'], params_env['t_drift'] = load_fun(params_env)
        # params_env['random_state'] = np.random.RandomState(seed=params_env['seed'])
        params_env['data_init_unlabelled_first'], _ = load_circle_init(n_train=2000, n_val=1000)

        print('good circle')

        # params_env['data_init_unlabelled_first'], params_env['data_arr'], params_env['data_init_unlabelled'], \
        # params_env['t_drift'] = load_circle_new(params_env)
        # print('circle_init_file_loaded')
    elif 'vib' in params_env['data_source']:
        params_env['data_arr'], params_env['data_init_unlabelled'], params_env['t_drift'] = load_fun(params_env)
        # params_env['random_state'] = np.random.RandomState(seed=params_env['seed'])
        params_env['data_init_unlabelled_first'], _ = load_vib_init(n_train=2000, n_val=1000)
        print('good vib')

    elif 'sine' in params_env['data_source']:
        params_env['data_arr'], params_env['data_init_unlabelled'], params_env['t_drift'] = load_fun(params_env)
        # params_env['random_state'] = np.random.RandomState(seed=params_env['seed'])
        params_env['data_init_unlabelled_first'], _ = load_sine_init(n_train=2000, n_val=1000)
        print('good sine')



    elif '23' in params_env['data_source']:

        params_env['data_arr'], params_env['data_init_unlabelled'],params_env['t_drift']= load_fun(params_env)
        params_env['data_init_unlabelled_first'],_=load_mnist_init(n_train=2000, n_val=1000)
        print('good 23 ')

    elif 'normal' in params_env['data_source']:
        params_env['data_arr'], params_env['data_init_unlabelled'], params_env['t_drift'] = load_fun(params_env)
        params_env['data_init_unlabelled_first'], _ = load_mnist_multinormal_init(n_train=2000, n_val=1000)
        print('good multinormal ')


    elif ('01' in params_env['data_source']) or ('multi' in params_env['data_source']):
        params_env['data_arr'], params_env['data_init_unlabelled'],_, params_env['t_drift']= load_fun(data_source=params_env['data_source'], num_init_unlabelled=2000,random_state=params_env['random_state'])
        params_env['data_init_unlabelled_first'],_=load_mnist_init(n_train=2000, n_val=1000)
        print('good 01/multi ')


    else:
        params_env['data_arr'], params_env['data_init_unlabelled'], _, params_env['t_drift'] = \
            load_fun(data_source=params_env['data_source'],num_init_unlabelled=params_env['unsupervised_win_size'],
                     random_state=params_env['random_state'])


    params_env['data_arr'] = params_env['data_arr'].astype(dtype='float32')
    params_env['data_init_unlabelled'] = params_env['data_init_unlabelled'].astype(dtype='float32')

    # common info for all datasets
    params_env['time_steps'] = params_env['data_arr'].shape[0]
    params_env['num_features'] = params_env['data_arr'].shape[1] - 1
    # params_env['num_classes'] = len(np.unique(params_env['data_arr'][:, -1]))
    data_arr = params_env['data_arr']
    if hasattr(data_arr, "iloc"):  # pandas DataFrame
        y = data_arr.iloc[:, -1].to_numpy()
    else:  # numpy array
        y = data_arr[:, -1]
    params_env['num_classes'] = len(np.unique(y))

    ###################
    # Settings: fixed #
    ###################
    # NOTE: Keep these parameters fixed to replicate the paper's results

    # rvus (suggested by its authors)
    params_env['rvus_active_threshold_update'] = 0.01
    params_env['rvus_active_delta'] = 1.0  # N(1, delta) - no randomisation if set to 0

    # budget spending (suggested by its authors)
    params_env['active_budget_window'] = 300
    params_env['active_budget_lambda'] = 1.0 - (1.0 / params_env['active_budget_window'])

    # prequential evaluation
    params_env['preq_fading_factor'] = 0.99

    ################
    # Output files #
    ################

    # file directory and names
    name_method = params_env['method']

    name_method += '_win' + str(params_env['unsupervised_win_size'])

    if params_env['method'] in ['ae', 'vae']:
        name_method += '_perc' + str(params_env['ae_threshold_percentile']) + \
                       '_freq' + str(params_env['unsupervised_win_size_update'])

    if params_env['method'] == 'vae':
        name_method += '_beta' + str(params_env['beta'])+\
                        'Pthre'+ str(params_env['Pthre'])+'adap'+str(params_env['adaptive'])


    if params_env['method'] == 'ae':
        name_method += '_avgstd' + str(params_env['noise_gaussian_avg']) + '_' + \
                       str(params_env['noise_gaussian_std'])+ '_' + \
                       str(params_env['reg_l1'])+\
                        'Pthre'+ str(params_env['Pthre'])+'adap'+str(params_env['adaptive'])

    out_name = '{}_{}'.format(params_env['data_source'], name_method)
    out_strategy=params_env['strategy']

    # files to store g-mean
    out_dir = 'exps/'
    filename_pred = os.path.join(os.getcwd(),  out_dir, out_name +out_strategy+str(params_env['num_epochs'])+'_'+str(params_env['lamda'])+'_'+str(params_env['Pthre'])+'_'+str(params_env['Dthre'])+'_'+str(params_env['incr'])+str(params_env['palarm'])+str(params_env['index'])+'_predictions.txt')
    create_file(filename_pred)

    filename_score = os.path.join(os.getcwd(),  out_dir, out_name +out_strategy+str(params_env['num_epochs'])+'_'+str(params_env['lamda'])+'_'+str(params_env['Pthre'])+'_'+str(params_env['Dthre'])+'_'+str(params_env['incr'])+str(params_env['palarm'])+str(params_env['index'])+'_score.txt')
    create_file(filename_score)

    filename_gmean = os.path.join(os.getcwd(),  out_dir, out_name +out_strategy+str(params_env['num_epochs'])+'_'+str(params_env['lamda'])+'_'+str(params_env['Pthre'])+'_'+str(params_env['Dthre'])+'_'+str(params_env['incr'])+str(params_env['palarm'])+str(params_env['index'])+'_preq_gmean.txt')
    create_file(filename_gmean)


    filename_recall = os.path.join(os.getcwd(),  out_dir, out_name +out_strategy+str(params_env['num_epochs'])+'_'+str(params_env['lamda'])+'_'+str(params_env['Pthre'])+'_'+str(params_env['Dthre'])+'_'+str(params_env['incr'])+str(params_env['palarm'])+str(params_env['index'])+'_preq_recall.txt')
    create_file(filename_recall)

    filename_specificity = os.path.join(os.getcwd(),  out_dir, out_name +out_strategy+str(params_env['num_epochs'])+'_'+str(params_env['lamda'])+'_'+str(params_env['Pthre'])+'_'+str(params_env['Dthre'])+'_'+str(params_env['incr'])+str(params_env['palarm'])+str(params_env['index'])+'_preq_specificity.txt')
    create_file(filename_specificity)

    #########
    # Start #
    #########

    for r in range(params_env['repeats']):
        print('Repetition: ', r)

        # model
        if params_env['strategy'] == 'ensemble_ae' or params_env['strategy'] == 'ensemble_incr'or params_env['strategy'] == 'ensemble_base' or params_env['strategy'] == 'ensemble_vae_DD'or params_env['strategy'] == 'ensemble_vae_new':
            params_env['fun_create_ae'] = create_ae
            params_env['fun_create_vae'] = create_vae
            params_env['fun_create_iforest'] = create_iforest
            params_env['fun_create_gmm'] = create_gmm
            params_env['fun_create_lof'] = create_lof
        else:
            if params_env['method'] == 'ae':
                params_env['fun_create_ae'] = create_ae
            elif params_env['method'] == 'vae':
                params_env['fun_create_vae'] = create_vae

            elif params_env['method'] == 'iforest':
                params_env['fun_create_iforest'] = create_iforest
            elif params_env['method'] == 'gmm':
                params_env['fun_create_gmm'] = create_gmm
            elif params_env['method'] == 'lof':
                params_env['fun_create_lof'] = create_lof

        # start
        _, preq_gmeans,recall_arr,specificity_arr,retrain_time,warn_time,alarm_time,pred_list,score_list = run(params_env)

        # store
        write_to_file(filename_gmean, preq_gmeans, retrain_time,warn_time,alarm_time)
        write_to_file(filename_recall, recall_arr, retrain_time, warn_time, alarm_time)
        write_to_file(filename_specificity, specificity_arr, retrain_time, warn_time, alarm_time)
        write_simple(filename_pred,pred_list)
        write_simple(filename_score, score_list)


