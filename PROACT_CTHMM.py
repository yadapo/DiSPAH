#!/usr/bin/env python
# coding: utf-8

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from fastdtw import dtw

import easydict

from functools import partial

import ALSdataread
from CTHMM_model import cthmm
from stratification import state_distance

from AnswerALS_CTHMM import score_subtotal

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
plt.rcParams['figure.subplot.left'] = 0.2


def main(args):
    print("+++++++++++++++++++++++++++++++++++++++")
    print('Loading ALSFRS-R scores from AnswerALS data')
    X, obs_timings, sequence_lengths, ALSusesubject_metadata_df, ALSusesubject_covar_df = ALSdataread.ALSdataprep_PROACT()
    print("+++++++++++++++++++++++++++++++++++++++")
    print('Estimating parameters using ALSFRS-R scores from AnswerALS data')
    num_samples = obs_timings.shape[0]
    progress_speed = np.zeros(args.num_samples)
    print('Estimating')
    start = time.time()
    ###################################
    # Fixed parameters #
    ref_data_dir = "./result1/"
    fixed_initial_state_prob = np.squeeze(np.load(ref_data_dir + "est_initial_state_prob_mat.npy"))
    fixed_emission_prob = np.load(ref_data_dir + "est_emission_prob_mat.npy")
    fixed_transition_prob_gene = np.load(ref_data_dir + "est_transition_prob_gene_mat.npy")
    ###################################
    (est_initial_state_prob, est_transition_prob_gene, est_emission_prob, est_progress_speed,
         log_likelihood_trajectory, optimal_paths) = cthmm(args.num_states, args.num_labels, num_samples, X,
                                                           obs_timings, sequence_lengths, early_stopping=args.early_stopping,
                                                           uniform_initialization=args.uniform_initialization,
                                                           uniform_speed=args.uniform_speed,
                                                           fixed_transition_prob_gene=fixed_transition_prob_gene,
                                                           fixed_emission_prob=fixed_emission_prob,
                                                           fixed_initial_state_prob=fixed_initial_state_prob)
    print("\nEM algorithm elapsed time:", time.time() - start)
    est_progress_speed_list = est_progress_speed.tolist()
    ALSusesubject_covar_df['est_progress_speed'] = est_progress_speed_list
    ALSusesubject_covar_df.to_csv('AnswerALS_covar_estimated_results.csv')
    ALSusesubject_metadata_df.to_csv('AnswerALS_metadata.csv')

    plt.figure(figsize=(5, 5))
    plt.plot(log_likelihood_trajectory, 'k-')
    plt.xlabel('The number of iterations')
    plt.ylabel('Log-likelihood')
    plt.savefig('Log-likelihood_tragectory.svg')
    plt.savefig('Log-likelihood_tragectory.jpg', dpi=300)
    plt.show()

    est_initial_state_prob = np.expand_dims(est_initial_state_prob, axis=1)
    plt.figure(figsize=(6,6))
    plt.imshow(est_initial_state_prob)
    plt.ylabel('State')
    plt.xticks([0])
    plt.colorbar()
    plt.savefig('estimated_initial_state_probability.svg')
    plt.savefig('estimated_initial_state_probability.jpg',dpi=300)
    plt.show()
    np.save('est_initial_state_prob_mat.npy', est_initial_state_prob)

    plt.figure(figsize=(6, 6))
    plt.imshow(est_transition_prob_gene, vmin=0, vmax=0.04)
    plt.xlabel('To state')
    plt.ylabel('From state')
    plt.colorbar()
    plt.savefig('estimated_transition_rate_matrix.svg')
    plt.savefig('estimated_transition_rate_matrix.jpg', dpi=300)
    plt.show()
    np.save('est_transition_prob_gene_mat.npy', est_transition_prob_gene)

    fig, axes = plt.subplots(3, 4, figsize=(12, 10))
    for ind in range(args.num_subscores):
        plot_row = np.int16(ind % 3)
        plot_col = np.int16(np.floor(ind / 3))
        im = axes[plot_row][plot_col].imshow(est_emission_prob[:, ind, :])
        axes[plot_row][plot_col].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    # Adding common labels
    fig.text(0.5, 0.04, 'Sub-score', ha='center', va='center', fontsize=28)
    fig.text(0.04, 0.5, 'State', ha='center', va='center', rotation='vertical', fontsize=28)
    plt.savefig('estimated_emission_prob_matrix.svg')
    plt.savefig('estimated_emission_prob_matrix.svg', dpi=300)
    plt.show()
    np.save('est_emission_prob_mat.npy', est_emission_prob)

    np.save('obs_timings.npy', obs_timings)
    np.save('sequence_lengths.npy', sequence_lengths)

    plt.figure(figsize=(16, 8))
    for samp_ind in range(num_samples):
        plt.plot(optimal_paths[samp_ind][0], optimal_paths[samp_ind][1], 'ko-')
    plt.xlim(0, 200)
    #plt.ylim(0, 50)
    plt.xlabel('Weeks from the first visit')
    plt.ylabel('State')
    plt.gca().invert_yaxis()
    plt.savefig('estimated_paths.svg', dpi=300)
    plt.savefig('estimated_paths.jpg', dpi=300)
    plt.show()

    state_traj = np.zeros([num_samples, obs_timings.shape[1]])
    unique_state_traj = []
    for samp_ind in range(num_samples):
        state_traj[samp_ind, :sequence_lengths[samp_ind]] = optimal_paths[samp_ind][1]
        unique_state_traj.append(np.unique(optimal_paths[samp_ind][1]))
    np.save('state_traj.npy', state_traj)

    
    #################################
    # Stratification based on the results from AnswerALS cohort.
    ##
    ref_estimated_results = pd.read_csv(ref_data_dir + "AnswerALS_covar_estimated_results.csv")
    ref_samp_clusters = ref_estimated_results['cluster']
    cluster_labels = np.unique(ref_samp_clusters)
    ref_state_traj = np.load(ref_data_dir + "state_traj.npy")
    ref_sequence_lengths = np.load(ref_data_dir + "sequence_lengths.npy")

    ref_num_samples = ref_estimated_results.shape[0]
    ref_unique_state_traj = []
    for ref_samp_ind in range(ref_num_samples):
        ref_unique_state_traj.append(np.unique(ref_state_traj[ref_samp_ind, 0:ref_sequence_lengths[ref_samp_ind]]))

    num_subscores = X.shape[2]
    partial_state_dist = partial(state_distance, num_subscores, est_emission_prob)
    state_traj_distance = np.zeros([num_samples, ref_num_samples])
    for samp_ind in range(num_samples):
        for ref_samp_ind in range(ref_num_samples):
            state_traj_distance[samp_ind, ref_samp_ind], _ = dtw(unique_state_traj[samp_ind],
                                                                  ref_unique_state_traj[ref_samp_ind],
                                                                  dist=partial_state_dist)
    samp_clusters = np.zeros([num_samples])
    for samp_ind in range(num_samples):
        nearest_ref_samp_ind = np.argmin(state_traj_distance[samp_ind])
        samp_clusters[samp_ind] = ref_samp_clusters[nearest_ref_samp_ind]

    plt.figure(figsize=(10, 5))
    for samp_ind in range(num_samples):
        if samp_clusters[samp_ind] == 1:
            color = 'C1'
        elif samp_clusters[samp_ind] == 2:
            color = 'C2'
        elif samp_clusters[samp_ind] == 3:
            color = 'C3'
        elif samp_clusters[samp_ind] == 4:
            color = 'C4'
        elif samp_clusters[samp_ind] == 5:
            color = 'C5'
        elif samp_clusters[samp_ind] == 6:
            color = 'C6'
        plt.plot(optimal_paths[samp_ind][0], optimal_paths[samp_ind][1], 'o-', color=color, linewidth=1, alpha=0.5, markersize=5)
    plt.xlim(0, 200)
    plt.xlabel('Weeks from the first visit')
    plt.ylabel('State')
    plt.gca().invert_yaxis()
    plt.savefig('estimated_paths_ref_cluster.svg', dpi=300)
    plt.savefig('estimated_paths_ref_cluster.jpg', dpi=300)
    plt.show()

    # Create the DataFrame
    PROACT_results_df = pd.DataFrame({
        'cluster': samp_clusters,
        'est_progress_speed': est_progress_speed
    })

    ######
    colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    # Estimated speed for each site of onset groups
    plt.figure(figsize=(5, 5))
    sns.swarmplot(x='cluster', y='est_progress_speed', data=PROACT_results_df, palette=colors, size=1.5)
    plt.xlabel('Cluster#')
    plt.ylabel('Estimated progression speed')
    plt.xticks([0,1,2,3,4,5],['1','2','3','4','5','6'])
    plt.ylim([-5.2,5.2])
    plt.savefig('AnswerALS_ref_cluster_est_progress_speed.svg', dpi=300)
    plt.savefig('AnswerALS_ref_cluster_est_progress_speed.jpg', dpi=300)
    plt.show()

    plt.figure(figsize=(5, 5))
    sns.violinplot(x='Riluzole', y='est_progress_speed', data=ALSusesubject_covar_df)
    plt.xticks([0, 1], ['Naive', 'Riluzole'])
    plt.ylabel('Estimated progression speed')
    plt.savefig('AnswerALS_riluzole_est_progress_speed.jpg', dpi=300)
    plt.show()
    #####################################



    #################################################
    #################################################
    subtotal_X = score_subtotal(X)
    num_domains = subtotal_X.shape[2]
    num_clusters = cluster_labels.shape[0]
    domain_names = ['Bulbar', 'Fine motor', 'Gross motor', 'Respiratory']

    #################################################
    num_representative_samples = 20
    cluster_representative_sample_ind = np.zeros([num_clusters, num_representative_samples], dtype=np.int16)
    cluster_representative_samples = np.zeros([num_clusters, num_representative_samples], dtype=np.int16)
    for cluster_ind in range(num_clusters):
        cluster_speed = est_progress_speed[samp_clusters == (cluster_ind+1)]
        cluster_samples = np.where(samp_clusters == (cluster_ind+1))[0]
        cluster_representative_sample_ind[cluster_ind] = np.argsort(np.abs(cluster_speed - np.mean(cluster_speed)))[:num_representative_samples]
        cluster_representative_samples[cluster_ind] = cluster_samples[cluster_representative_sample_ind[cluster_ind]]

    #############
    cluster_labels = np.int8(np.unique(samp_clusters))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    plt.subplots_adjust(wspace=0.5, hspace=0)  # Increase these values for wider spacing

    for cluster_ind in reversed(range(num_clusters)):
        if cluster_ind + 1 == 1:
            color = 'C1'
        elif cluster_ind + 1 == 2:
            color = 'C2'
        elif cluster_ind + 1 == 3:
            color = 'C3'
        elif cluster_ind + 1 == 4:
            color = 'C4'
        elif cluster_ind + 1 == 5:
            color = 'C5'
        elif cluster_ind + 1 == 6:
            color = 'C6'
        cluster_rep_X = subtotal_X[cluster_representative_samples[cluster_ind]]
        cluster_rep_sequenth_length = sequence_lengths[cluster_representative_samples[cluster_ind]]

        plot_ind = 0
        for domain_ind in range(num_domains):
            for another_domain_ind in range(domain_ind+1, num_domains):
                plot_col = np.int16(plot_ind % 3)
                plot_row = np.int16(np.floor(plot_ind / 3))

                for samp_ind in range(cluster_representative_samples.shape[1]):
                    axes[plot_row][plot_col].plot(cluster_rep_X[samp_ind, :cluster_rep_sequenth_length[samp_ind], domain_ind],
                                              cluster_rep_X[samp_ind, :cluster_rep_sequenth_length[samp_ind], another_domain_ind], '.-', color=color, alpha=0.8)
                axes[plot_row][plot_col].set_xlabel(domain_names[domain_ind])
                axes[plot_row][plot_col].set_xlim([-0.5, 12.5])
                axes[plot_row][plot_col].set_xticks([0, 6, 12])
                axes[plot_row][plot_col].set_ylabel(domain_names[another_domain_ind])
                axes[plot_row][plot_col].set_ylim([-0.5, 12.5])
                axes[plot_row][plot_col].set_yticks([0, 6, 12])
                axes[plot_row][plot_col].set_aspect('equal')
                plot_ind = plot_ind + 1

    plt.savefig('cluster_domain_score_pathway.svg')
    plt.savefig('cluster_domain_score_pathway.jpg', dpi=300)
    plt.show()

    for cluster_ind in range(num_clusters):
        if cluster_ind + 1 == 1:
            color = 'C1'
        elif cluster_ind + 1 == 2:
            color = 'C2'
        elif cluster_ind + 1 == 3:
            color = 'C3'
        elif cluster_ind + 1 == 4:
            color = 'C4'
        elif cluster_ind + 1 == 5:
            color = 'C5'
        elif cluster_ind + 1 == 6:
            color = 'C6'

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.subplots_adjust(wspace=0.5, hspace=0.3)

        plot_ind = 0
        for domain_ind in range(num_domains):
            for another_domain_ind in range(domain_ind + 1, num_domains):
                plot_col = np.int16(plot_ind % 3)
                plot_row = np.int16(np.floor(plot_ind / 3))
                cluster_rep_X = subtotal_X[cluster_representative_samples[cluster_ind]]
                cluster_rep_sequenth_length = sequence_lengths[cluster_representative_samples[cluster_ind]]
                for samp_ind in range(cluster_representative_samples.shape[1]):
                    axes[plot_row][plot_col].plot(
                        cluster_rep_X[samp_ind, :cluster_rep_sequenth_length[samp_ind], domain_ind],
                        cluster_rep_X[samp_ind, :cluster_rep_sequenth_length[samp_ind], another_domain_ind], '.-',
                        color=color, alpha=0.8)
                axes[plot_row][plot_col].set_xlabel(domain_names[domain_ind])
                axes[plot_row][plot_col].set_xlim([-0.5, 12.5])
                axes[plot_row][plot_col].set_xticks([0, 6, 12])
                axes[plot_row][plot_col].set_ylabel(domain_names[another_domain_ind])
                axes[plot_row][plot_col].set_ylim([-0.5, 12.5])
                axes[plot_row][plot_col].set_yticks([0, 6, 12])
                axes[plot_row][plot_col].set_aspect('equal')
                plot_ind = plot_ind + 1

        fig.suptitle(f"Cluster {cluster_labels[cluster_ind]}", fontsize=16)

        out_svg = f"cluster_{cluster_labels[cluster_ind]}_domain_score_pathway.svg"
        out_jpg = f"cluster_{cluster_labels[cluster_ind]}_domain_score_pathway.jpg"
        plt.savefig(out_svg)
        plt.savefig(out_jpg, dpi=300)
        plt.show()
        plt.close(fig)

    ##########################


    ## representative trajectories of clusters for each sub-total sub-scores
    fig, axes = plt.subplots(1, 4, figsize=(16, 6))
    for cluster_ind in reversed(range(num_clusters)):
        if cluster_ind+1 == 1:
            color = 'C1'
        elif cluster_ind+1 == 2:
            color = 'C2'
        elif cluster_ind+1 == 3:
            color = 'C3'
        elif cluster_ind+1 == 4:
            color = 'C4'
        elif cluster_ind+1 == 5:
            color = 'C5'
        elif cluster_ind+1 == 6:
            color = 'C6'
        cluster_rep_X = subtotal_X[cluster_representative_samples[cluster_ind]]
        cluster_rep_T = obs_timings[cluster_representative_samples[cluster_ind]]
        cluster_rep_sequenth_length = sequence_lengths[cluster_representative_samples[cluster_ind]]
        for domain_ind in range(num_domains):
            plot_row = 1
            plot_col = domain_ind
            for samp_ind in range(cluster_representative_samples.shape[1]):
                axes[plot_col].plot(cluster_rep_T[samp_ind, :cluster_rep_sequenth_length[samp_ind]],
                                              cluster_rep_X[samp_ind, :cluster_rep_sequenth_length[samp_ind], domain_ind], '.-', color=color, alpha=0.8)
            axes[plot_col].set_ylim([-0.5, 12.5])
            axes[plot_col].set_title(domain_names[domain_ind])
    fig.text(0.55, 0.06, 'Time', ha='center', va='center', fontsize=24)
    fig.text(0.12, 0.5, 'Sub-score', ha='center', va='center', rotation='vertical', fontsize=24)
    plt.savefig('cluster_representative_subtotalsubscore_trajectory.svg')
    plt.savefig('cluster_representative_subtotalsubscore_trajectory.jpg', dpi=300)
    plt.show()

    for cluster_ind in range(num_clusters):
        fig, axes = plt.subplots(1, 4, figsize=(16, 6))

        if cluster_ind+1 == 1:
            color = 'C1'
        elif cluster_ind+1 == 2:
            color = 'C2'
        elif cluster_ind+1 == 3:
            color = 'C3'
        elif cluster_ind+1 == 4:
            color = 'C4'
        elif cluster_ind+1 == 5:
            color = 'C5'
        elif cluster_ind+1 == 6:
            color = 'C6'
        cluster_rep_X = subtotal_X[cluster_representative_samples[cluster_ind]]
        cluster_rep_T = obs_timings[cluster_representative_samples[cluster_ind]]
        cluster_rep_sequenth_length = sequence_lengths[cluster_representative_samples[cluster_ind]]

        for domain_ind in range(num_domains):
            plot_row = 1
            plot_col = domain_ind
            for samp_ind in range(cluster_representative_samples.shape[1]):
                axes[plot_col].plot(cluster_rep_T[samp_ind, :cluster_rep_sequenth_length[samp_ind]],
                                              cluster_rep_X[samp_ind, :cluster_rep_sequenth_length[samp_ind], domain_ind], '.-', color=color, alpha=0.8)
            axes[plot_col].set_ylim([-0.5, 12.5])
            axes[plot_col].set_title(domain_names[domain_ind])
        fig.text(0.55, 0.06, 'Time', ha='center', va='center', fontsize=24)
        fig.text(0.12, 0.5, 'Sub-score', ha='center', va='center', rotation='vertical', fontsize=24)
        out_svg = f"cluster_{cluster_labels[cluster_ind]}_representative_subtotalsubscore_trajectory.svg"
        out_jpg = f"cluster_{cluster_labels[cluster_ind]}_representative_subtotalsubscore_trajectory.jpg"
        plt.savefig(out_svg)
        plt.savefig(out_jpg, dpi=300)
        plt.show()
        plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hidden Markov Model")
    parser.add_argument("--num-states", default=3, type=int)
    parser.add_argument("--num-labels", default=10, type=int)
    parser.add_argument("--num-data", default=500, type=int)
    parser.add_argument("-n", "--num-samples", nargs="?", default=1000, type=int)
    args = easydict.EasyDict({
        "num_states": 6,
        "num_subscores": 12,
        "num_labels": 5,  # currently not use this argument.
        "num_data": 10,
        "num_samples": 500,
        "uniform_speed": False,
        "early_stopping": True,
        "uniform_initialization": True,
    })

    main(args)
