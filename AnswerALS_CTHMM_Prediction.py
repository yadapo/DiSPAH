#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from jax import random

from fastdtw import dtw

from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import ALSdataread
from CTHMM_model import cthmm
from stratification import state_distance

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


def evaluate_regression_performance(test_y, pred_y):
    # Mean Absolute Error
    mae = mean_absolute_error(test_y, pred_y)
    print(f'Mean Absolute Error (MAE): {mae:.4f}')

    # Mean Squared Error
    mse = mean_squared_error(test_y, pred_y)
    print(f'Mean Squared Error (MSE): {mse:.4f}')

    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    print(f'Root Mean Squared Error (RMSE): {rmse:.4f}')

    # R-squared
    r2 = r2_score(test_y, pred_y)
    print(f'R-squared (R2): {r2:.4f}')

    # Residual Plot
    residuals = test_y - pred_y
    plt.figure(figsize=(10, 6))
    sns.residplot(x=pred_y, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.savefig('regression_results_residual_plot.jpg', dpi=300)
    plt.savefig('regression_results_residual_plot.svg')
    plt.show()

    # Write metrics to a txt file
    with open("prediction_results_speed.txt", "w") as f:
        f.write(f'Mean Absolute Error (MAE): {mae:.4f}\n')
        f.write(f'Mean Squared Error (MSE): {mse:.4f}\n')
        f.write(f'Root Mean Squared Error (RMSE): {rmse:.4f}\n')
        f.write(f'R-squared (R2): {r2:.4f}\n')


def evaluate_clusterpred_performance(test_y, pred_y):
    # Accuracy
    accuracy = accuracy_score(test_y, pred_y)
    print(f'Accuracy: {accuracy:.4f}')

    # Precision, Recall, F1-Score (weighted to account for imbalance)
    precision = precision_score(test_y, pred_y, average='weighted')
    recall = recall_score(test_y, pred_y, average='weighted')
    f1 = f1_score(test_y, pred_y, average='weighted')

    print(f'Precision (weighted): {precision:.4f}')
    print(f'Recall (weighted): {recall:.4f}')
    print(f'F1-Score (weighted): {f1:.4f}')

    # Classification report
    class_report = classification_report(test_y, pred_y)
    print("\nClassification Report:")
    print(class_report)

    # Confusion Matrix
    cls_label = ["1","2","3","4","5","6"]
    cm = confusion_matrix(test_y, pred_y)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=cls_label, yticklabels=cls_label)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('prediction_results_cluster_confusion_map.jpg', dpi=300)
    plt.savefig('prediction_results_cluster_confusion_map.svg')
    plt.show()

    # Write metrics to a txt file
    with open("prediction_results_cluster.txt", "w") as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision (weighted): {precision:.4f}\n')
        f.write(f'Recall (weighted): {recall:.4f}\n')
        f.write(f'F1-Score (weighted): {f1:.4f}\n\n')
        f.write("Classification Report:\n")
        f.write(class_report + "\n")
        f.write("Confusion Matrix:\n")
        np.savetxt(f, cm, fmt='%d')


def wrapper_function(args, X, obs_timings, sequence_lengths, ALSusesubject_metadata_df, ALSusesubject_covar_df):
    cvset_ind, unpair_index, pair_train_index, pair_test_index = args
    return parallelized_function(cvset_ind, unpair_index, pair_train_index, pair_test_index, X, obs_timings, sequence_lengths, ALSusesubject_metadata_df, ALSusesubject_covar_df)


def parallelized_function(cvset_ind, unpair_index, pair_train_index, pair_test_index, X=None, obs_timings=None,
                          sequence_lengths=None, ALSusesubject_metadata_df=None, ALSusesubject_covar_df=None):
    print('Cross-validation set: ' + str(cvset_ind))
    ######################################################################3333
    # Preparation of a dataset
    #########
    num_states = 6
    num_labels = 5  # each sub-score can take 0-4 points.
    num_subscores = X.shape[2]

    unpair_X = X[unpair_index]
    unpair_obs_timings = obs_timings[unpair_index]
    unpair_sequence_lengths = sequence_lengths[unpair_index]
    unpair_num_samples = unpair_obs_timings.shape[0]
    unpair_metadata_df = ALSusesubject_metadata_df.iloc[unpair_index]
    unpair_covar_df = ALSusesubject_covar_df.iloc[unpair_index]
    pair_train_X = X[pair_train_index]
    pair_train_obs_timings = obs_timings[pair_train_index]
    pair_train_sequence_lengths = sequence_lengths[pair_train_index]
    pair_train_num_samples = pair_train_obs_timings.shape[0]
    pair_train_metadata_df = ALSusesubject_metadata_df.iloc[pair_train_index]
    pair_train_covar_df = ALSusesubject_covar_df.iloc[pair_train_index]
    pair_test_X = X[pair_test_index]
    if pair_test_index.shape[0] == 1:
        pair_test_obs_timings = np.expand_dims(obs_timings[pair_test_index],0)[0]
    else:
        pair_test_obs_timings = obs_timings[pair_test_index]
    pair_test_sequence_lengths = sequence_lengths[pair_test_index]
    pair_test_num_samples = pair_test_obs_timings.shape[0]
    pair_test_metadata_df = ALSusesubject_metadata_df.iloc[pair_test_index]
    pair_test_covar_df = ALSusesubject_covar_df.iloc[pair_test_index]

    train_X = np.concatenate([unpair_X, pair_train_X])
    train_obs_timings = np.concatenate([unpair_obs_timings, pair_train_obs_timings])
    train_sequence_lengths = np.concatenate([unpair_sequence_lengths, pair_train_sequence_lengths])
    train_num_samples = train_obs_timings.shape[0]
    train_metadata_df = pd.concat([unpair_metadata_df, pair_train_metadata_df], axis=0)
    train_covar_df = pd.concat([unpair_covar_df, pair_train_covar_df], axis=0)

    ######################################################################3333
    # Learning with the train data
    #########
    rng_key_traintest = random.PRNGKey(0)
    rng_key_train, rng_key_test = random.split(rng_key_traintest, 2)

    (est_initial_state_prob, est_transition_prob_gene, est_emission_prob, train_est_progress_speed, log_likelihood_trajectory,
     train_optimal_paths) = cthmm(num_states, num_labels, train_num_samples, train_X, train_obs_timings,
                            train_sequence_lengths, rng_key=rng_key_train, early_stopping=True, iteration_num=20, uniform_initialization=True)
    train_est_progress_speed_list = train_est_progress_speed.tolist()
    train_covar_df['est_progress_speed'] = train_est_progress_speed_list

    setname = 'cvset' + str(cvset_ind)

    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(log_likelihood_trajectory.shape[0]) + 1, log_likelihood_trajectory, 'k-')
    plt.xlabel('The number of iterations')
    plt.ylabel('Log-likelihood')
    plt.savefig('Log-likelihood_tragectory_' + setname + '.svg')
    plt.show()

    est_initial_state_prob_plot = np.expand_dims(est_initial_state_prob, axis=1)
    plt.figure(figsize=(6, 6))
    plt.imshow(est_initial_state_prob_plot, vmin=0)
    plt.ylabel('State')
    plt.xticks([0])
    plt.colorbar()
    plt.savefig('estimated_initial_state_probability_' + setname + '.svg')
    plt.show()

    plt.figure(figsize=(6, 6))
    plt.imshow(est_transition_prob_gene, vmin=0)
    plt.xlabel('To state')
    plt.ylabel('From state')
    plt.colorbar()
    plt.savefig('estimated_transition_rate_matrix_' + setname + '.svg')
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
    plt.savefig('estimated_emission_prob_matrix_' + setname + '.svg')
    plt.show()


    #################
    # Clustering training samples

    train_state_traj = np.zeros([train_num_samples, obs_timings.shape[1]])
    train_unique_state_traj = []

    for samp_ind in range(train_num_samples):
        train_state_traj[samp_ind, :train_sequence_lengths[samp_ind]] = train_optimal_paths[samp_ind][1]
        train_unique_state_traj.append(np.unique(train_optimal_paths[samp_ind][1]))

    num_subscores = X.shape[2]

    partial_state_dist = partial(state_distance, num_subscores, est_emission_prob)

    #####
    num_clusters = 6
    ref_major_states = np.zeros([num_clusters, num_clusters])
    ref_major_states[0] = [1,1,0,0,0,0]
    ref_major_states[1] = [0,0,1,1,1,0]
    ref_major_states[2] = [0,0,1,1,1,1]
    ref_major_states[3] = [1,1,0,0,1,1]
    ref_major_states[4] = [1,0,1,1,0,1]
    ref_major_states[5] = [1,1,1,1,0,0]

    train_cluster_labels = np.zeros([train_num_samples])
    for samp_ind in range(train_num_samples):
        samp_states = np.zeros([num_states])
        for state_ind in train_unique_state_traj[samp_ind]:
            samp_states[state_ind] = 1
        if samp_states[0] == 0 and samp_states[1] ==0:
            if samp_states[5] == 0:
                train_cluster_labels[samp_ind] = 2
            else:
                train_cluster_labels[samp_ind] = 3
        else:
            if samp_states[2] == 0 and samp_states[3] ==0:
                if samp_states[4] == 0 and samp_states[5] ==0:
                    train_cluster_labels[samp_ind] = 1
                else:
                    train_cluster_labels[samp_ind] = 4                
            else:
                if samp_states[4] == 0 and samp_states[5] ==0:
                    train_cluster_labels[samp_ind] = 6
                else:
                    train_cluster_labels[samp_ind] = 5

    #####
    
    plt.figure(figsize=(10, 5))
    for samp_ind in range(train_num_samples):
        if train_cluster_labels[samp_ind] == 1:
            color = 'C1'
        elif train_cluster_labels[samp_ind] == 2:
            color = 'C2'
        elif train_cluster_labels[samp_ind] == 3:
            color = 'C3'
        elif train_cluster_labels[samp_ind] == 4:
            color = 'C4'
        elif train_cluster_labels[samp_ind] == 5:
            color = 'C5'
        elif train_cluster_labels[samp_ind] == 6:
            color = 'C6'
        plt.plot(train_optimal_paths[samp_ind][0], train_optimal_paths[samp_ind][1], 'o-', color=color, linewidth=1, alpha=0.5,
                 markersize=5)
    plt.xlim(0, 200)
    plt.xlabel('Weeks from the first visit')
    plt.ylabel('State')
    plt.gca().invert_yaxis()
    plt.savefig('estimated_paths_cluster' + setname + '.svg', dpi=300)
    plt.show()

    train_covar_df['cluster'] = train_cluster_labels

    train_covar_df['ALSFRSR_init_bulbar'] = np.sum(train_X[:, 0, 0:3], axis=1)
    train_covar_df['ALSFRSR_init_finemotor'] = np.sum(train_X[:, 0, 3:6], axis=1)
    train_covar_df['ALSFRSR_init_grossmotor'] = np.sum(train_X[:, 0, 6:9], axis=1)
    train_covar_df['ALSFRSR_init_respiratory'] = np.sum(train_X[:, 0, 9:12], axis=1)


    ##########################################
    # Estimation with the test data

    (_, _, _, est_pair_test_progress_speed, pair_test_log_likelihood_trajectory, pair_test_optimal_paths) = cthmm(num_states, num_labels, pair_test_num_samples, pair_test_X, pair_test_obs_timings, pair_test_sequence_lengths, rng_key=rng_key_test, early_stopping=True, iteration_num=20,
                                                           uniform_initialization=True, for_test=True,
                                                           fixed_transition_prob_gene=est_transition_prob_gene,
                                                           fixed_emission_prob=est_emission_prob,
                                                           fixed_initial_state_prob=est_initial_state_prob)
    est_pair_test_progress_speed_list = est_pair_test_progress_speed.tolist()
    pair_test_covar_df['est_progress_speed'] = est_pair_test_progress_speed_list


    #################
    # Clustering test
    pair_test_state_traj = np.zeros([pair_test_num_samples, pair_test_obs_timings.shape[1]])
    pair_test_unique_state_traj = []
    for samp_ind in range(pair_test_num_samples):
        pair_test_state_traj[samp_ind, :pair_test_sequence_lengths[samp_ind]] = pair_test_optimal_paths[samp_ind][1]
        pair_test_unique_state_traj.append(np.unique(pair_test_optimal_paths[samp_ind][1]))
      
    pair_test_state_traj_distance = np.zeros([pair_test_num_samples, train_num_samples])
    for samp_ind in range(pair_test_num_samples):
        for train_samp_ind in range(train_num_samples):
            pair_test_state_traj_distance[samp_ind, train_samp_ind], _ = dtw(pair_test_unique_state_traj[samp_ind],
                                                                  train_unique_state_traj[train_samp_ind],
                                                                  dist=partial_state_dist)
    pair_test_cluster_labels = np.zeros([pair_test_num_samples])
    for samp_ind in range(pair_test_num_samples):
        nearest_train_samp_ind = np.argmin(pair_test_state_traj_distance[samp_ind])
        pair_test_cluster_labels[samp_ind] = train_cluster_labels[nearest_train_samp_ind]

    plt.figure(figsize=(10, 5))
    for samp_ind in range(pair_test_num_samples):
        if pair_test_cluster_labels[samp_ind] == 1:
            color = 'C1'
        elif pair_test_cluster_labels[samp_ind] == 2:
            color = 'C2'
        elif pair_test_cluster_labels[samp_ind] == 3:
            color = 'C3'
        elif pair_test_cluster_labels[samp_ind] == 4:
            color = 'C4'
        elif pair_test_cluster_labels[samp_ind] == 5:
            color = 'C5'
        elif pair_test_cluster_labels[samp_ind] == 6:
            color = 'C6'          
        plt.plot(pair_test_optimal_paths[samp_ind][0], pair_test_optimal_paths[samp_ind][1], 'o-', color=color, linewidth=1, alpha=0.5, markersize=5)
    plt.xlim(0, 200)
    plt.xlabel('Weeks from the first visit')
    plt.ylabel('State')
    plt.gca().invert_yaxis()
    plt.savefig('test_estimated_paths_cluster' + setname + '.svg', dpi=300)
    plt.show()

    pair_test_covar_df['cluster'] = pair_test_cluster_labels

    pair_test_covar_df['ALSFRSR_init_bulbar'] = np.sum(pair_test_X[:, 0, 0:3], axis=1)
    pair_test_covar_df['ALSFRSR_init_finemotor'] = np.sum(pair_test_X[:, 0, 3:6], axis=1)
    pair_test_covar_df['ALSFRSR_init_grossmotor'] = np.sum(pair_test_X[:, 0, 6:9], axis=1)
    pair_test_covar_df['ALSFRSR_init_respiratory'] = np.sum(pair_test_X[:, 0, 9:12], axis=1)

    

    #####################

    ######
    colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    # Estimated speed for each site of onset groups    
    plt.figure(figsize=(5, 5))
    sns.swarmplot(x='cluster', y='est_progress_speed', data=pair_test_covar_df, palette=colors, size=1.5)
    plt.xlabel('Cluster#')
    plt.ylabel('Estimated progression speed')
    plt.savefig('test_cluster_est_progress_speed' + setname + '.svg', dpi=300)
    plt.show()

    ############################################
    # Prediction of the progression speed and the custer from iPSC transcriptome

    traintest_covar_df = pd.concat([train_covar_df, pair_test_covar_df])

    transcriptome_als_scaled_df, traintest_dataset_df = ALSdataread.ALS_transcriptome_dataprep(traintest_covar_df)
    traintest_dataset_df['C9orf72 mutation'] = np.int8((traintest_dataset_df['C9orf72 repeat length']) > 100) | np.int8(traintest_dataset_df['Mutations'] == 'C9orf72')
    traintest_dataset_df['ATXN2 mutation'] = np.int8(traintest_dataset_df['ATXN2 repeat length'] >= 27) | np.int8(traintest_dataset_df['Mutations']=='ATXN2')
    traintest_dataset_df['SOD1 mutation'] = np.int8(traintest_dataset_df['Mutations']=='SOD1')
    traintest_dataset_df['Riluzole'] = np.int8(traintest_dataset_df['Riluzole'])
    traintest_dataset_df = traintest_dataset_df.drop(['Site of Onset', 'ALSFRS-R Progression Slope', 'C9orf72 repeat length', 'ATXN2 repeat length', 'Mutations'], axis=1)
    train_dataset_df = traintest_dataset_df.iloc[:pair_train_num_samples]
    print("Does the train dataset contain NaN elements?")
    print(train_dataset_df.isnull().values.any())
    test_dataset_df = traintest_dataset_df.iloc[pair_train_num_samples:]
    print("Does the test dataset contain NaN elements?")
    print(test_dataset_df.isnull().values.any())

    if ~test_dataset_df.isnull().values.any():
        train_dataset_df.dropna(inplace=True)

         ###########################
        # Prediction #
        train_features_df = train_dataset_df.drop(['est_progress_speed', 'cluster'], axis=1)
        test_features_df = test_dataset_df.drop(['est_progress_speed', 'cluster'], axis=1)

        ##############
        # Prediction of estimated progress speed
        train_est_speed = train_dataset_df['est_progress_speed']
        test_est_speed = test_dataset_df['est_progress_speed']

        ##############
        # Prediction of clusterstest_dataset_df.isnull().values.any()
        train_cluster = train_dataset_df['cluster']
        test_cluster = test_dataset_df['cluster']

        #--
        basic_columns = ['C9orf72 mutation', 'SOD1 mutation', 'ALSFRSR_init_bulbar','ALSFRSR_init_finemotor', 'ALSFRSR_init_grossmotor','ALSFRSR_init_respiratory']
        nonuse_columns = ['Sex', 'Age At Symptom Onset', 'Site of Onset', 'ALSFRS-R Progression Slope', 'C9orf72 repeat length', 'ATXN2 repeat length', 'Mutations', 'Riluzole']
        train_omicsdata_df = train_features_df[[col for col in train_features_df.columns if col not in basic_columns+nonuse_columns]]
        train_otherdata_df = train_features_df[basic_columns]
        test_omicsdata_df= test_features_df[[col for col in test_features_df.columns if col not in basic_columns+nonuse_columns]]
        test_otherdata_df = test_features_df[basic_columns]

        # For the estimated progression cluster #########################
        p_results = {}
        coef_results = {}
        for geneprot_ind, geneprot_name in enumerate(train_omicsdata_df.columns):
            X = sm.add_constant(train_dataset_df[['Sex', 'Age At Symptom Onset', 'C9orf72 mutation']])
            X[geneprot_name] = train_dataset_df[geneprot_name]
            y = train_dataset_df['est_progress_speed']
            model = sm.OLS(y, X).fit()
            p_results[geneprot_name] = model.pvalues[geneprot_name]
            coef_results[geneprot_name] = model.params[geneprot_name]

        geneprots = list(p_results.keys())
        pvalues = np.asarray([p_results[geneprot] for geneprot in geneprots])
        coefs = np.asarray([coef_results[geneprot] for geneprot in geneprots])
        results_df = pd.DataFrame({
            'genes/proteins': geneprots,
            'pvalues': pvalues,
            'coefs': coefs
        })
        results_df = results_df.sort_values(by='pvalues')
        print(results_df.head(10))

        # Add the phenotype column to the DataFrame
        ##### Adust this parameter for SFig. 7 #########
        num_genes = 5
        ################################################
        train_total_df = pd.concat([train_otherdata_df, train_omicsdata_df[results_df['genes/proteins'].values[:num_genes]]], axis=1)

        # Add the phenotype column to the DataFrame
        test_total_df = pd.concat(
            [test_otherdata_df, test_omicsdata_df[results_df['genes/proteins'].values[:num_genes]]], axis=1)

        ## for est_speed
        print('##################')
        print('Prediction of progression speeds')

        ridge = Ridge()
        param_grid = {'alpha': [0.01, 0.1, 1.0, 10, 100]}
        grid_search_reg = GridSearchCV(estimator=ridge, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)

        ##
        grid_search_reg.fit(train_total_df, train_est_speed)
        # Obtain the optimal parameters and the model
        best_model_reg = grid_search_reg.best_estimator_
        print("the optimal params for regression:", grid_search_reg.best_params_)

        pred_test_est_speed = best_model_reg.predict(test_total_df)

        print("progression speed")
        print(test_est_speed)
        print(pred_test_est_speed)

        # Learning and Prediction ####################3
        print('##################')
        print('Prediction of progression clusters')

        svm = SVC(kernel='linear', class_weight='balanced')

        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100]  # Regularization parameter
        }

        grid_search_classify = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='accuracy', cv=5)
        ####
        train_combined_df = train_total_df.copy()
        train_combined_df['est_speed'] = train_est_speed
        test_combined_df = test_total_df.copy()
        test_combined_df['est_speed'] = pred_test_est_speed
        ####
        grid_search_classify.fit(train_combined_df, train_cluster)
        best_model_classify = grid_search_classify.best_estimator_
        print("the optimal params for classification:", grid_search_classify.best_params_)
        print("Best Cross-Validation Accuracy:", grid_search_classify.best_score_)

        # Evaluate performance in test data.
        pred_test_cluster = best_model_classify.predict(test_combined_df)
        print("clusters")
        print(test_cluster)
        print(pred_test_cluster)


    else:
        test_est_speed = test_dataset_df['est_progress_speed']
        pred_test_est_speed = np.nan
        test_cluster = test_dataset_df['cluster']
        pred_test_cluster = np.nan

    results = np.zeros([2, 2, pair_test_num_samples])
    # progress speed
    results[0,0] = test_est_speed
    results[0,1] = pred_test_est_speed
    # cluster
    results[1,0] = test_cluster
    results[1,1] = pred_test_cluster

    return results


def main():
    print("+++++++++++++++++++++++++++++++++++++++")
    print('Loading ALSFRS-R scores from AnswerALS data')
    
    X, obs_timings, sequence_lengths, ALSusesubject_metadata_df, ALSusesubject_covar_df = ALSdataread.ALSdataprep()
    print('Loading iPSC transcriptome data from AnswerALS data')
    transcriptome_als_scaled_df, ALSusesubject_covar_transcriptome_df = ALSdataread.ALS_transcriptome_dataprep(ALSusesubject_covar_df)
    pair_sample_index = np.where(ALSusesubject_covar_df['GUID'].isin(ALSusesubject_covar_transcriptome_df.index))[0]
    unpair_sample_index = np.where(~ALSusesubject_covar_df['GUID'].isin(ALSusesubject_covar_transcriptome_df.index))[0]
    print("+++++++++++++++++++++++++++++++++++++++")

    loo = LeaveOneOut()
    args_list = []
    cvset_ind = np.int8(0)
    for train_index, test_index in loo.split(pair_sample_index):
        args_list.append((cvset_ind, unpair_sample_index, pair_sample_index[train_index], pair_sample_index[test_index]))
        cvset_ind = cvset_ind + 1
    args_list = args_list[::-1]

    # Create a partial function with the additional fixed arguments
    func = partial(wrapper_function, X=X, obs_timings=obs_timings, sequence_lengths=sequence_lengths, ALSusesubject_metadata_df=ALSusesubject_metadata_df, ALSusesubject_covar_df=ALSusesubject_covar_df)

    # Modify how you call executor.map
    max_workers = 30
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use a list comprehension to correctly combine the arguments
        partial_samp_results = list(executor.map(func, args_list))

    cv_samp_num = partial_samp_results[0].shape[2]
    test_samp_num = pair_sample_index.shape[0]
    all_samp_results = np.zeros([2,2,test_samp_num])
    cum_samp_index = 0
    for res_ind, partial_samp_result in enumerate(partial_samp_results):
        all_samp_results[:, :, cum_samp_index:cum_samp_index+partial_samp_result.shape[2]] = partial_samp_result
        cum_samp_index = cum_samp_index + partial_samp_result.shape[2]

    samp_res_nonnan = ~np.isnan(np.sum(np.sum(all_samp_results, axis=0), axis=0))
    all_samp_results = all_samp_results[:,:,samp_res_nonnan]

    np.save('prediction_results.npy', all_samp_results)

    plt.figure(figsize=(6, 6))
    plt.scatter(all_samp_results[0][0],all_samp_results[0][1])
    plt.plot(np.array([[-4, -4],[4,4]]),np.array([[-4, -4],[4,4]]),'k--')
    ticks = np.arange(-4, 5, 2)  # -4, -2, 0, 2, 4
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.xlabel('Estimated progression speed')
    plt.ylabel('Predicted progression speed')
    plt.savefig('prediction_results_progress_speed.jpg', dpi=300)
    plt.savefig('prediction_results_progress_speed.svg')
    plt.show()

    evaluate_clusterpred_performance(all_samp_results[1][0], all_samp_results[1][1])
    evaluate_regression_performance(all_samp_results[0][0], all_samp_results[0][1])


if __name__ == "__main__":
    main()

