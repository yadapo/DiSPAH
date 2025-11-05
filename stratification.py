#!/usr/bin/env python
# coding: utf-8


import matplotlib.pyplot as plt
import numpy as np

from fastdtw import dtw
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import curve_fit

from functools import partial
from scipy.stats import wasserstein_distance


def state_distance(num_subscores, est_emission_prob, state_a, state_b):
    tmp_distance = np.zeros([num_subscores])
    for subscore_ind in range(num_subscores):
        tmp_distance[subscore_ind] = wasserstein_distance(est_emission_prob[np.int8(state_a), subscore_ind], est_emission_prob[np.int8(state_b), subscore_ind])
    distance = np.mean(tmp_distance)
    return distance

def stratification(obs_timings=None, X=None, sequence_lengths=None, est_emission_prob=None, est_progress_speed=None, optimal_paths=None, unique_state_traj=None, ALSusesubject_covar_df=None):
    num_samples = X.shape[0]
    num_subscores = X.shape[2]
    num_states = est_emission_prob.shape[0]

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
    dendrogram(Z)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Sample index")
    plt.ylabel("Distance")
    plt.savefig('trajectory_dendrogram.svg', dpi=300)
    plt.savefig('trajectory_dendrogram.jpg', dpi=300)
    plt.show()

    t = 0.7 * max(Z[:, 2])
    clusters = fcluster(Z, t, criterion="distance")

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
        plt.plot(optimal_paths[samp_ind][0], optimal_paths[samp_ind][1], 'o-', color=color, alpha=0.5)
    plt.xlim(0, 200)
    #plt.ylim(0, 50)
    plt.xlabel('Weeks from the first visit')
    plt.ylabel('State')
    plt.gca().invert_yaxis()
    plt.savefig('estimated_paths_cluster.svg', dpi=300)
    plt.savefig('estimated_paths_cluster.jpg', dpi=300)
    plt.show()

    ALSusesubject_covar_df['cluster'] = clusters
    ALSusesubject_covar_df.to_csv('AnswerALS_covar_estimated_results.csv')

    #################################################
    num_clusters = np.unique(clusters).shape[0]
    num_cluster_samples = np.zeros([num_clusters])
    cluster_states = np.zeros([num_clusters, num_states])
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
        prev_state = None
        for state in major_cluster_states_list[cluster_ind]:
            if prev_state is not None:
                plt.plot(np.array([prev_state, state+deviation, state+deviation]), np.array([prev_state+deviation, prev_state+deviation, state]), '-', linewidth=2, color=color, alpha=0.5)
            prev_state = state
        for state in major_cluster_states_list[cluster_ind]:
            plt.plot(state, state, 'o', markersize=markersize, color='white', alpha=1)
            plt.plot(state, state, 'o', markersize=markersize, color=color, alpha=0.8)
    for state in range(num_states):
        plt.plot(state, state, 'o', markersize=5, color='white', alpha=1)
    plt.xlabel("state")
    plt.ylabel("state")
    plt.xticks(range(num_states))
    plt.yticks(range(num_states))
    plt.xlim([-1, num_states])
    plt.ylim([-1, num_states])
    plt.gca().invert_yaxis()
    plt.savefig('cluster_major_path.svg', dpi=300)
    plt.savefig('cluster_major_path.jpg', dpi=300)
    plt.show()
    #################################################

    #################################################
    for cluster_ind in range(num_clusters):
        cluster_est_emission_prob = est_emission_prob[major_cluster_states_list[cluster_ind]]
        fig, axes = plt.subplots(3, 4, figsize=(12, 10))
        for subscore_ind in range(num_subscores):
            plot_row = np.int16(subscore_ind % 3)
            plot_col = np.int16(np.floor(subscore_ind / 3))
            im = axes[plot_row][plot_col].imshow(cluster_est_emission_prob[:, subscore_ind, :], vmin=0, vmax=1)
            axes[plot_row][plot_col].axis('off')
        fig.colorbar(im, ax=axes.ravel().tolist())
        # Adding common labels
        fig.text(0.5, 0.04, 'Sub-score', ha='center', va='center', fontsize=28)
        fig.text(0.04, 0.5, 'State', ha='center', va='center', rotation='vertical', fontsize=28)
        plt.savefig('cluster' + str(cluster_ind + 1) + '_estimated_emission_prob_matrix.svg')
        plt.savefig('cluster' + str(cluster_ind + 1) + '_estimated_emission_prob_matrix.jpg', dpi=300)
        plt.show()

    #################################################
    #################################################
    # depict representative trajectories of clusters for each sub-score.
    def sigmoid(x, alpha, tau):
        return (4 / (1 + np.exp(-alpha * (x - tau))))
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for cluster_ind in range(num_clusters):
        cluster_speed = est_progress_speed[clusters == (cluster_ind+1)]
        cluster_samples = np.where(clusters == (cluster_ind+1))[0]
        cluster_representative_sample_ind = np.argsort(np.abs(cluster_speed - np.mean(cluster_speed)))[:5]
        cluster_representative_samples = cluster_samples[cluster_representative_sample_ind]
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
        cluster_rep_X = X[cluster_representative_samples]
        cluster_rep_T = obs_timings[cluster_representative_sample_ind]
        cluster_rep_sequenth_length = sequence_lengths[cluster_representative_sample_ind]
        for subscore_ind in range(num_subscores):
            aggregate_x = None
            aggregate_t = None
            for samp_ind in range(cluster_representative_samples.shape[0]):
                tmp_x = cluster_rep_X[samp_ind, :cluster_rep_sequenth_length[samp_ind], subscore_ind]
                tmp_t = np.squeeze(cluster_rep_T[samp_ind, :cluster_rep_sequenth_length[samp_ind]])
                if aggregate_x is None:
                    aggregate_x = tmp_x
                    aggregate_t = tmp_t
                else:
                    aggregate_x = np.concatenate([aggregate_x, tmp_x])
                    aggregate_t = np.concatenate([aggregate_t, tmp_t])
            popt, _ = curve_fit(sigmoid, aggregate_t, aggregate_x, bounds=([-0.2, -100], [0, 400]))
            print(popt)
            t_smooth = np.linspace(0, 200, 200)
            x_smooth = sigmoid(t_smooth, *popt)

            plot_row = np.int16(subscore_ind % 3)
            plot_col = np.int16(np.floor(subscore_ind / 3))
            axes[plot_row, plot_col].plot(aggregate_t, aggregate_x, '.', color=color)
            axes[plot_row, plot_col].plot(t_smooth, x_smooth, '-', color=color, linewidth=4, alpha=0.7)
    fig.text(0.5, 0.06, 'Time', ha='center', va='center', fontsize=28)
    fig.text(0.12, 0.5, 'Sub-score', ha='center', va='center', rotation='vertical', fontsize=28)
    plt.savefig('cluster_representative_subscore_trajectory.svg')
    plt.savefig('cluster_representative_subscore_trajectory.jpg', dpi=300)
    plt.show()
    #################################################

