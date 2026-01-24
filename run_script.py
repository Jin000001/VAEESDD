from main import main


for lr in [0.0001]:
    for ds in ['mnist_23_severe']:
        lst_ds = [

            {'repeats': 1, 'data_source': ds,
             'strategy': 'baseline',
             'method': 'vae',
             'unsupervised_win_size': 200, 'ae_threshold_percentile': 80, 'unsupervised_win_size_update': 1.0,
         'mov_win_size':1000,
             'beta': 1.0, 'noise_gaussian_avg': 0, 'noise_gaussian_std': 0, 'lamda': 0, 'reg_l1': 0, 'Pthre': 2, 'Dthre': 'nodd','incr': 'yes','Esnum': 10,'palarm':0.001,'index':'esdd',
             'adaptive': 'yes', 'num_epochs': 200,'lr':lr},
        ]

        for d in lst_ds:
            main(d)
