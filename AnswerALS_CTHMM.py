#!/usr/bin/env python
# coding: utf-8

import argparse
import time

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from jax.scipy.linalg import expm

from fastdtw import dtw
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform

import easydict

import ALSdataread
from CTHMM_model import cthmm
from stratification import state_distance
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
plt.rcParams['figure.subplot.left'] = 0.2


def score_subtotal(X):
    num_samples = X.shape[0]
    num_observations = X.shape[1]
    num_subscores = X.shape[2]

    subtotal_X = np.zeros((num_samples, num_observations, 4))
    subtotal_X[:, :, 0] = np.sum(X[:, :, 0:3], axis=2)
    subtotal_X[:, :, 1] = np.sum(X[:, :, 3:6], axis=2)
    subtotal_X[:, :, 2] = np.sum(X[:, :, 6:9], axis=2)
    subtotal_X[:, :, 3] = np.sum(X[:, :, 9:], axis=2)

    return subtotal_X

def main(args):
    print("+++++++++++++++++++++++++++++++++++++++")
    print('Loading ALSFRS-R scores from AnswerALS data')
    X, obs_timings, sequence_lengths, ALSusesubject_metadata_df, ALSusesubject_covar_df = ALSdataread.ALSdataprep()
    print("+++++++++++++++++++++++++++++++++++++++")
    print('Estimating parameters using ALSFRS-R scores from AnswerALS data')
    num_samples = obs_timings.shape[0]
    progress_speed = np.zeros(args.num_samples)
    print('Estimating')
    start = time.time()

    (est_initial_state_prob, est_transition_prob_gene, est_emission_prob, est_progress_speed,log_likelihood_trajectory, optimal_paths) = cthmm(args.num_states, args.num_labels, num_samples, X, obs_timings, sequence_lengths, early_stopping=args.early_stopping, uniform_initialization=args.uniform_initialization, uniform_speed=args.uniform_speed, iteration_num=20)

    print("\nEM algorithm elapsed time:", time.time() - start)
    est_progress_speed_list = est_progress_speed.tolist()
    ALSusesubject_covar_df['est_progress_speed'] = est_progress_speed_list
    ALSusesubject_covar_df.to_csv('AnswerALS_covar_estimated_results.csv')
    ALSusesubject_metadata_df.to_csv('AnswerALS_metadata.csv')

    fast_trasition_prob = expm(est_transition_prob_gene*np.exp(1.5)*5)
    np.savetxt("fast_transition_prob.txt", fast_trasition_prob, fmt="%.4f")
    slow_trasition_prob = expm(est_transition_prob_gene*np.exp(-0.5)*5)
    np.savetxt("slow_transition_prob.txt", slow_trasition_prob, fmt="%.4f")

    plt.figure(figsize=(6, 6))
    plt.plot(log_likelihood_trajectory, 'k-')
    plt.xticks([0,2,4,6,8,10])
    plt.xlabel('The number of iterations')
    plt.ylabel('Log-likelihood')
    plt.savefig('Log-likelihood_tragectory.svg')
    plt.savefig('Log-likelihood_tragectory.jpg', dpi=300)
    plt.show()


    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05)
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_scatter)

    ax_scatter.scatter(ALSusesubject_covar_df['ALSFRS-R Progression Slope'], ALSusesubject_covar_df['est_progress_speed'])
    ax_scatter.set_xlabel('Progression slope')
    ax_scatter.set_ylabel('Estimated progression speed')

    # Histograms
    ax_histx.hist(ALSusesubject_covar_df['ALSFRS-R Progression Slope'], bins=30)
    ax_histy.hist(ALSusesubject_covar_df['est_progress_speed'], bins=30, orientation='horizontal')

    # Clean up tick labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    plt.savefig('slope_estimated_speed.svg')
    plt.savefig('slope_estimated_speed.jpg', dpi=300)
    plt.show()


    est_initial_state_prob = np.expand_dims(est_initial_state_prob, axis=1)
    plt.figure(figsize=(4.2,3.6))
    plt.imshow(est_initial_state_prob,vmin=0, vmax=1, cmap='YlOrRd')
    plt.ylabel('State')
    plt.yticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    plt.xticks([0])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Probability')
    plt.savefig('estimated_initial_state_probability.svg')
    plt.savefig('estimated_initial_state_probability.jpg', dpi=300)
    plt.show()
    np.save('est_initial_state_prob_mat.npy', est_initial_state_prob)

    plt.figure(figsize=(6, 3.6))
    plt.imshow(est_transition_prob_gene, cmap='RdYlBu_r', vmin=-0.05, vmax=0.05)
    plt.xlabel('To state')
    plt.ylabel('From state')
    plt.xticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    plt.yticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Transition rate')
    plt.savefig('estimated_transition_rate_matrix.svg')
    plt.savefig('estimated_transition_rate_matrix.jpg', dpi=300)
    plt.show()
    np.save('est_transition_prob_gene_mat.npy', est_transition_prob_gene)

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for ind in range(args.num_subscores):
        plot_row = np.int16(ind % 3)
        plot_col = np.int16(np.floor(ind / 3))
        im = axes[plot_row][plot_col].imshow(est_emission_prob[:, ind, :], vmin=0, vmax=1, cmap='YlOrRd')
        axes[plot_row][plot_col].axis('off')
    cbar = fig.colorbar(im, ax=axes.ravel().tolist())
    cbar.ax.set_ylabel('Probability', fontsize=28)
    cbar.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.ax.tick_params(labelsize=28)
    # Adding common labels
    fig.text(0.5, 0.1, 'Sub-score', ha='center', va='center', fontsize=28)
    fig.text(0.1, 0.5, 'State', ha='center', va='center', rotation='vertical', fontsize=28)
    plt.savefig('estimated_emission_prob_matrix.svg')
    plt.savefig('estimated_emission_prob_matrix.jpg', dpi=300)
    plt.show()
    np.save('est_emission_prob_mat.npy', est_emission_prob)

    np.save('obs_timings.npy', obs_timings)
    np.save('sequence_lengths.npy', sequence_lengths)

    plt.figure(figsize=(8, 4))
    for samp_ind in range(num_samples):
        plt.plot(optimal_paths[samp_ind][0], optimal_paths[samp_ind][1], 'ko-', linewidth=0.5, alpha=0.4, markersize=3)
    plt.xlim(0, 200)
    plt.xlabel('Weeks from the first visit')
    plt.ylabel('State')
    plt.yticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
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
    unique_state_traj_onevalue = np.zeros(num_samples)
    for samp_ind in range(num_samples):
        for state_ind in range(est_transition_prob_gene.shape[0]):
            if np.isin(state_ind, unique_state_traj[samp_ind]):
                unique_state_traj_onevalue[samp_ind] += (1 << state_ind)
    ALSusesubject_covar_df['unique_states'] = unique_state_traj_onevalue
    ALSusesubject_covar_df.to_csv('AnswerALS_covar_estimated_results.csv')

    num_subscores = X.shape[2]

    partial_state_dist = partial(state_distance, num_subscores, est_emission_prob)
    state_traj_distance = np.zeros([num_samples, num_samples])
    for samp_ind in range(num_samples):
        for pair_samp_ind in range(samp_ind+1, num_samples):
            state_traj_distance[samp_ind, pair_samp_ind], _ = dtw(unique_state_traj[samp_ind], unique_state_traj[pair_samp_ind], dist=partial_state_dist)
            state_traj_distance[pair_samp_ind, samp_ind] = state_traj_distance[samp_ind, pair_samp_ind]

    condensed_state_traj_distance = squareform(state_traj_distance)

    # Perform hierarchical clustering using the linkage function
    Z = linkage(condensed_state_traj_distance, method='ward')

    # Plot the dendrogram
    plt.figure(figsize=(8, 6))
    dendrogram(Z, color_threshold=0.5)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.savefig('trajectory_dendrogram.svg', dpi=300)
    plt.savefig('trajectory_dendrogram.jpg', dpi=300)
    plt.show()

    clusters = fcluster(Z, t=6, criterion='maxclust')

    plt.figure(figsize=(10, 5))
    for samp_ind in range(num_samples):
        if clusters[samp_ind] == 1:
            color = 'C1'
        elif clusters[samp_ind] == 2:
            color = 'C2'
        elif clusters[samp_ind] == 3:
            color = 'C3'
        elif clusters[samp_ind] == 4:
            color = 'C4'
        elif clusters[samp_ind] == 5:
            color = 'C5'
        elif clusters[samp_ind] == 6:
            color = 'C6'
        plt.plot(optimal_paths[samp_ind][0], optimal_paths[samp_ind][1], 'o-', color=color, linewidth=1, alpha=0.5, markersize=5)
    plt.xlim(0, 200)
    plt.xlabel('Weeks from the first visit')
    plt.ylabel('State')
    plt.yticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    plt.gca().invert_yaxis()
    plt.savefig('estimated_paths_cluster.svg', dpi=300)
    plt.savefig('estimated_paths_cluster.jpg', dpi=300)
    plt.show()

    ALSusesubject_covar_df['cluster'] = clusters
    ALSusesubject_covar_df.to_csv('AnswerALS_covar_estimated_results.csv')

    #################################################
    num_clusters = np.unique(clusters).shape[0]
    num_cluster_samples = np.zeros([num_clusters])
    cluster_states = np.zeros([num_clusters, args.num_states])
    major_cluster_states_list = []
    plt.figure(figsize=(5, 5))
    for cluster_ind in range(num_clusters):
        for samp_ind in np.where(clusters-1==cluster_ind)[0]:
            cluster_states[cluster_ind,unique_state_traj[samp_ind]] += 1
        num_cluster_samples[cluster_ind] = len(np.where(clusters-1==cluster_ind)[0])
        major_cluster_states_list.append(np.where(cluster_states[cluster_ind] > (num_cluster_samples[cluster_ind]/10))[0])
        if cluster_ind+1 == 1:
            color = 'C1'
            markersize = 30
            deviation = -0.16
        elif cluster_ind+1 == 2:
            color = 'C2'
            markersize = 25
            deviation = -0.08
        elif cluster_ind+1 == 3:
            color = 'C3'
            markersize = 20
            deviation = 0
        elif cluster_ind+1 == 4:
            color = 'C4'
            markersize = 15
            deviation = 0.08
        elif cluster_ind+1 == 5:
            color = 'C5'
            markersize = 10
            deviation = 0.16
        elif cluster_ind+1 == 6:
            color = 'C6'
            markersize = 7
            deviation = 0.24
        prev_state = None
        for state in major_cluster_states_list[cluster_ind]:
            if prev_state is not None:
                plt.plot(np.array([prev_state, state+deviation, state+deviation]), np.array([prev_state+deviation, prev_state+deviation, state]), '-', linewidth=2, color=color, alpha=0.5)
            prev_state = state
        for state in major_cluster_states_list[cluster_ind]:
            plt.plot(state, state, 'o', markersize=markersize, color='white', alpha=1)
            plt.plot(state, state, 'o', markersize=markersize, color=color, alpha=0.8)
    for state in range(args.num_states):
        plt.plot(state, state, 'o', markersize=5, color='white', alpha=1)
    plt.xlabel("State")
    plt.ylabel("State")
    plt.xticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    plt.yticks([0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6])
    plt.xticks(range(args.num_states))
    plt.yticks(range(args.num_states))
    plt.xlim([-1, args.num_states])
    plt.ylim([-1, args.num_states])
    plt.gca().invert_yaxis()
    plt.savefig('cluster_major_path.svg', dpi=300)
    plt.savefig('cluster_major_path.jpg', dpi=300)
    plt.show()
    #################################################

    #################################################
    for cluster_ind in range(num_clusters):
        cluster_est_emission_prob = est_emission_prob[major_cluster_states_list[cluster_ind]]
        fig, axes = plt.subplots(3, 4, figsize=(10, 8))
        for subscore_ind in range(args.num_subscores):
            plot_row = np.int16(subscore_ind % 3)
            plot_col = np.int16(np.floor(subscore_ind / 3))
            im = axes[plot_row][plot_col].imshow(cluster_est_emission_prob[:, subscore_ind, :], vmin=0, vmax=1)
            axes[plot_row][plot_col].axis('off')
        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.ax.set_ylabel('Probability', fontsize=28)
        cbar.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
        cbar.ax.tick_params(labelsize=28)
        # Adding common labels
        fig.text(0.5, 0.1, 'Sub-score', ha='center', va='center', fontsize=28)
        fig.text(0.1, 0.5, 'State', ha='center', va='center', rotation='vertical', fontsize=28)
        plt.savefig('cluster' + str(cluster_ind + 1) + '_estimated_emission_prob_matrix.svg')
        plt.savefig('cluster' + str(cluster_ind + 1) + '_estimated_emission_prob_matrix.jpg', dpi=300)
        plt.show()

    #################################################
    #################################################
    subtotal_X = score_subtotal(X)
    num_domains = subtotal_X.shape[2]
    domain_names = ['Bulbar', 'Fine motor', 'Gross motor', 'Respiratory']


    #################################################
    num_representative_samples = 20
    cluster_representative_sample_ind = np.zeros([num_clusters, num_representative_samples], dtype=np.int16)
    cluster_representative_samples = np.zeros([num_clusters, num_representative_samples], dtype=np.int16)
    for cluster_ind in range(num_clusters):
        cluster_speed = est_progress_speed[clusters == (cluster_ind+1)]
        cluster_samples = np.where(clusters == (cluster_ind+1))[0]
        cluster_representative_sample_ind[cluster_ind] = np.argsort(np.abs(cluster_speed - np.mean(cluster_speed)))[:num_representative_samples]
        cluster_representative_samples[cluster_ind] = cluster_samples[cluster_representative_sample_ind[cluster_ind]]

    #############
    cluster_labels = np.unique(clusters)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    plt.subplots_adjust(wspace=0.5, hspace=0)  

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
        "uniform_initialization":True,
    })

    main(args)
