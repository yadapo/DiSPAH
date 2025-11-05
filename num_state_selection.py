#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

import ALSdataread
from CTHMM_model import cthmm, cthmm_likelihood

from concurrent.futures import ProcessPoolExecutor
from functools import partial

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 22
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=MEDIUM_SIZE)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.subplot.bottom'] = 0.15
plt.rcParams['figure.subplot.left'] = 0.25


def wrapper_function(args, X, obs_timings, sequence_lengths):
    num_states, cvset_ind, train_index, val_index = args
    return parallelized_function(num_states, cvset_ind, train_index, val_index, X, obs_timings, sequence_lengths)


def parallelized_function(num_states, cvset_ind, train_index, val_index, X=None, obs_timings=None, sequence_lengths=None):
    print('Cross-validation set: ' + str(cvset_ind))
    num_labels = 5 # each sub-score can take 0-4 points.
    num_subscores = X.shape[2]
    train_X = X[train_index]
    train_obs_timings = obs_timings[train_index]
    train_sequence_lengths = sequence_lengths[train_index]
    train_num_samples = train_obs_timings.shape[0]
    val_X = X[val_index]
    val_obs_timings = obs_timings[val_index]
    val_sequence_lengths = sequence_lengths[val_index]
    val_num_samples = val_obs_timings.shape[0]

    (est_initial_state_prob, est_transition_prob_gene, est_emission_prob, est_progress_speed, log_likelihood_trajectory,
     optimal_paths) = cthmm(num_states, num_labels, train_num_samples, train_X, train_obs_timings,
                            train_sequence_lengths, early_stopping=True, iteration_num=20,
                            uniform_initialization=True)

    val_log_likelihood = cthmm_likelihood(num_states, val_num_samples, val_X, val_obs_timings, val_sequence_lengths, est_initial_state_prob, est_transition_prob_gene, est_emission_prob, est_progress_speed)
    
    setname = 'numstates'+str(num_states)+'_cvset'+str(cvset_ind)
    
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(log_likelihood_trajectory.shape[0]) + 1, log_likelihood_trajectory, 'k-')
    plt.xlabel('The number of iterations')
    plt.ylabel('Log-likelihood')
    plt.savefig('Log-likelihood_tragectory_'+setname+'.svg')
    plt.show()

    est_initial_state_prob = np.expand_dims(est_initial_state_prob, axis=1)
    plt.figure(figsize=(6,6))
    plt.imshow(est_initial_state_prob, vmin=0)
    plt.ylabel('State')
    plt.xticks([0])
    plt.colorbar()
    plt.savefig('estimated_initial_state_probability_'+setname+'.svg')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(est_transition_prob_gene, vmin=0)
    plt.xlabel('To state')
    plt.ylabel('From state')
    plt.colorbar()
    plt.savefig('estimated_transition_rate_matrix_'+setname+'.svg')
    plt.show()

    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    for ind in range(num_subscores):
        plot_row = np.int16(ind % 3)
        plot_col = np.int16(np.floor(ind / 3))
        im = axes[plot_row][plot_col].imshow(est_emission_prob[:, ind, :], vmin=0, vmax=1)
        axes[plot_row][plot_col].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    # Adding common labels
    fig.text(0.5, 0.04, 'Sub-score', ha='center', va='center', fontsize=28)
    fig.text(0.04, 0.5, 'State', ha='center', va='center', rotation='vertical', fontsize=28)
    plt.savefig('estimated_emission_prob_matrix_'+setname+'.svg')
    plt.show()

    return val_log_likelihood
    

def main():
    print("+++++++++++++++++++++++++++++++++++++++")
    print('Loading ALSFRS-R scores from AnswerALS data')

    X, obs_timings, sequence_lengths, ALSusesubject_metadata_df, ALSusesubject_covar_df = ALSdataread.ALSdataprep()
    print("+++++++++++++++++++++++++++++++++++++++")
    print('Validation the optimal number of states')
    num_samples = obs_timings.shape[0]

    #################
    num_state_candidates = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=np.int8)
    num_candidates = num_state_candidates.shape[0]
    num_cv_set = np.int8(10)

    args_list = []
    for ns_ind, num_states in enumerate(num_state_candidates):
        print("+++++++++++++++++++++++++++++++++++++++")
        print('Number of states: '+str(num_states))
        kf = KFold(n_splits=num_cv_set, shuffle=False)
        cvset_ind = np.int8(0)
        for train_index, val_index in kf.split(np.arange(0, num_samples)):
            args_list.append((num_states, cvset_ind, train_index, val_index))
            cvset_ind = cvset_ind+1

    # Create a partial function with the additional fixed arguments
    func = partial(wrapper_function, X=X, obs_timings=obs_timings, sequence_lengths=sequence_lengths)

    # Modify how you call executor.map
    max_workers = 28
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use a list comprehension to correctly combine the arguments
        val_log_likelihood_list = list(executor.map(func, args_list))

    val_log_likelihood_array = np.zeros([num_candidates, num_cv_set])
    for res_ind, val_log_likelihood in enumerate(val_log_likelihood_list):
        row = np.int8(np.floor(res_ind / num_cv_set))
        col = np.int8(res_ind % num_cv_set)
        val_log_likelihood_array[row, col] = val_log_likelihood[0]  


    plt.figure(figsize=(6, 5))
    for ns_ind in range(num_candidates):
        plt.plot(np.repeat(num_state_candidates[ns_ind], num_cv_set),val_log_likelihood_array[ns_ind, :], 'o', color='black')
    mean_val_ll = np.nanmean(val_log_likelihood_array, axis=1)
    plt.plot(num_state_candidates, mean_val_ll, '-', color='black', linewidth=2.5)
    plt.xlabel('The number of states')
    plt.ylabel('Log-likelihood')
    plt.savefig('validation_log_likelihood_various_states.svg')
    plt.savefig('validation_log_likelihood_various_states.jpg', dpi=300)
    plt.show()

    # Combine into a DataFrame
    df = pd.DataFrame({
        'Number of States': num_state_candidates,
        'Mean Log-likelihood': mean_val_ll
    })

    # Save to CSV
    df.to_csv('mean_val_ll_output.csv', index=False)

if __name__ == "__main__":
    main()
