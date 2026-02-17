To execute the script:run_script.py
parameter setting:
'strategy':'ensemble_incr'(VAE++ES),'ensemble_base'(VAEES),'ensemble_vae_DD'(VAE++ESDD),'baseline'(baseline),'hybrid'(stream++DD)
'method'：'ae', 'vae', 'iforest', 'gmm', 'lof'
'unsupervised_win_size': drift detection window size
'ae_threshold_percentile': fixed anomaly detection threshold percentile
'unsupervised_win_size_update': percentile of replacement of moving window during update
'mov_win_size': moving window size
'beta': VAE loss function beta
'noise_gaussian_avg': DAE mean
'noise_gaussian_std': DAE std
'lamda': for Gmean 
'reg_l1': L1 regularization coefficient for AE
'Pthre': anomaly voting threshold
'Dthre': drift detection voting threshold
'incr': 'yes'/'no', to determine if it is incremental
'Esnum': number of ensemble members
'palarm': drift alarm threshold
'index':choose ensemble ('esdd') / single detector ('onedd')
'adaptive': 'yes'/'no', to determine if anomaly detection is adaptive
'num_epochs': number of epoch for model update
