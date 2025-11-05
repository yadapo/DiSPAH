#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import optimize as optim

from jax import random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.linalg import expm

import numpyro.distributions as dist


def cthmm(num_states, num_labels, num_samples, X, T, sequence_lengths, rng_key=random.PRNGKey(0), iteration_num = 20, early_stopping = False, uniform_initialization=True, uniform_speed=False, for_test=False, fixed_transition_prob_gene=None, fixed_emission_prob=None, fixed_initial_state_prob=None):
    rng_key_initial_state, rng_key_transition, rng_key_emission = random.split(rng_key, 3)
    print("--------------cthmm---------------")
    num_subscores = X.shape[2]

    deltaT_unique_tmp = np.array([])
    for samp_ind in range(0, num_samples):
        deltaT = T[samp_ind, 1:sequence_lengths[samp_ind]] - T[samp_ind, 0:sequence_lengths[samp_ind] - 1]
        deltaT_unique_tmp = np.concatenate((deltaT_unique_tmp, np.unique(deltaT)))
    print(deltaT_unique_tmp)
    total_deltaT_unique = np.asarray(np.unique(deltaT_unique_tmp), dtype=np.int16)
    num_total_deltaT_unique = total_deltaT_unique.shape[0]
    print(total_deltaT_unique)

    # initial parameter
    print('#initial parameter#')
    if fixed_transition_prob_gene is None:
        transition_prob_elem = dist.Uniform(0.008, 0.01).sample(
            key=rng_key_transition, sample_shape=(num_states, num_states)
    
        )
        transition_prob_gene = np.zeros((num_states,num_states))
        for from_state_ind in range(0,num_states):
            if from_state_ind != num_states-1:
                for to_state_ind in range(from_state_ind + 1, num_states):
                    if uniform_initialization:
                        transition_prob_gene[from_state_ind, to_state_ind] = 0.01
                    else:
                        transition_prob_gene[from_state_ind, to_state_ind] = transition_prob_elem[from_state_ind, to_state_ind]
        for state_ind in range(0, num_states):
            transition_prob_gene[state_ind,state_ind] = -np.sum(transition_prob_gene[state_ind,:])
            # Summation of each row in transition prob generator is zeros in MJP.
    else:
        transition_prob_gene = fixed_transition_prob_gene
    print('transition rate matrix:')
    print(transition_prob_gene)

    print('emission probability matrix:')
    if fixed_emission_prob is None:
        emission_prob = np.ones([num_states, num_subscores, num_labels])
        emission_prob = emission_prob / num_labels
        print(emission_prob)
        log_emission_prob = np.log(emission_prob)
    else:
        print(fixed_emission_prob)
        log_emission_prob = np.log(fixed_emission_prob)
        

    print('initial state probability:')
    if fixed_initial_state_prob is None:
        initial_z_prob = np.zeros([num_states, ])
        num_initial_states = np.int8(np.floor(num_states / 2))
        if uniform_initialization:
            initial_z_prob[:num_initial_states] = np.ones([num_initial_states]) * 1/num_initial_states
        else:
            initial_z_prob[:num_initial_states] = dist.Dirichlet(jnp.array([np.ones(num_initial_states)*100])).sample(key=rng_key_initial_state, sample_shape=(1,))
        print(initial_z_prob)
        log_initial_z_prob = np.log(initial_z_prob)
    else:
        print(fixed_initial_state_prob)
        log_initial_z_prob = np.log(fixed_initial_state_prob)

    # initialization for the progress speed parameters
    print('progress speed:')
    progress_speed = np.zeros([num_samples])
    print(progress_speed)


    # EM Algorithm for continuous-time HMM cf.[Liu et al., 2015]
    log_likelihood_trajectory = np.zeros([iteration_num])
    for iter_ind in range(0, iteration_num):
        if iter_ind is not 0:
            prev_log_likelihood = log_likelihood
        else:
            prev_log_likelihood = np.log(0)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("EM interation: "+str(iter_ind))
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        log_gamma_array = np.full((num_samples, np.max(sequence_lengths), num_states), -np.inf, dtype=np.float64)
        log_xi_array = np.full((num_samples, np.max(sequence_lengths) - 1, num_states, num_states), -np.inf, dtype=np.float64)
        log_expected_stayingtime = np.full((num_samples, num_states), -np.inf, dtype=np.float64)
        log_expected_transition_times = np.full((num_samples, num_states, num_states), -np.inf, dtype=np.float64)
        total_log_count_table = np.full((num_total_deltaT_unique, num_states, num_states), -np.inf, dtype=np.float64)

        for samp_ind in range(0, num_samples):
            deltaT = T[samp_ind, 1:sequence_lengths[samp_ind]] - T[samp_ind, 0:sequence_lengths[samp_ind]-1]
            deltaT_unique = np.unique(deltaT)
            num_deltaT_unique = deltaT_unique.shape[0]

            transition_prob = np.ones([num_deltaT_unique, num_states, num_states])
            for delT_ind, delT in enumerate(deltaT_unique):
                tmp_mat = np.exp(progress_speed[samp_ind]) * delT * transition_prob_gene
                transition_prob[delT_ind] = expm(tmp_mat)
            transition_prob = jnp.stack(transition_prob)
            log_transition_prob = np.log(transition_prob)

            X_samp = X[samp_ind, :sequence_lengths[samp_ind]]

            (log_likelihood_tmp, log_gamma_array[samp_ind, 0:sequence_lengths[samp_ind], :], log_xi_array[samp_ind, 0:sequence_lengths[samp_ind]-1, :, :]) = forward_backward_algorithm(X_samp, log_initial_z_prob, deltaT, log_transition_prob, log_emission_prob)
            if samp_ind == 0:
                log_likelihood = log_likelihood_tmp
            else:
                log_likelihood = log_likelihood_tmp + log_likelihood
            log_count_table = compute_counttable(num_states, deltaT, log_xi_array[samp_ind])
            for delT_ind, delT in enumerate(deltaT_unique):
                if samp_ind == 1:
                    total_log_count_table[total_deltaT_unique == delT, :, :] = log_count_table[delT_ind, :, :]
                else:
                    total_log_count_table[total_deltaT_unique == delT, :, :] = np.log(np.exp(total_log_count_table[total_deltaT_unique == delT, :, :])+np.exp(log_count_table[delT_ind, :, :]))
            (log_expected_stayingtime[samp_ind], log_expected_transition_times[samp_ind]) = expectation_expm_method(num_states, deltaT_unique, progress_speed[samp_ind], transition_prob_gene, log_transition_prob, log_count_table)
        print('## log-likelihood ##')
        print(log_likelihood)
        print('####')
        if early_stopping:
            if (log_likelihood - prev_log_likelihood)/np.abs(log_likelihood) < 0.001:
                log_likelihood_trajectory = log_likelihood_trajectory[:iter_ind]
                break
        log_likelihood_trajectory[iter_ind] = log_likelihood
        ####

        print('**************Updated Parameters***************')
        if fixed_initial_state_prob is None:
            # Update the parameter (initial state probability)
            tmp1 = np.full((num_samples, num_states), -np.inf, dtype=np.float64) 
            tmp2 = np.full((num_samples), -np.inf, dtype=np.float64) 
            for samp_ind in range(0, num_samples):
                tmp1[samp_ind] = log_gamma_array[samp_ind, 0, :]
                tmp2[samp_ind] = logsumexp(log_gamma_array[samp_ind, 0, :])
            log_initial_z_prob = logsumexp(tmp1, axis=0) - logsumexp(tmp2, axis=0)

            initial_z_prob = np.exp(log_initial_z_prob)
            initial_z_prob[np.int8(np.floor(num_states / 2)):] = np.zeros([initial_z_prob[np.int8(np.floor(num_states / 2)):].shape[0]])
            initial_z_prob = initial_z_prob / np.sum(initial_z_prob)
            log_initial_z_prob = np.log(initial_z_prob)
        print('current initial z prob.')
        print(jnp.exp(log_initial_z_prob))

        prev_transition_prob_gene = transition_prob_gene.copy()
        if fixed_transition_prob_gene is None:
            # Update the parameter (transition prob generator matrix)
            for from_state_ind in range(0,num_states):
                if from_state_ind + 1 != num_states:
                    for to_state_ind in range(from_state_ind+1, num_states):
                        transition_prob_gene[from_state_ind, to_state_ind] = np.exp(logsumexp(log_expected_transition_times[:, from_state_ind, to_state_ind]) - logsumexp(progress_speed + log_expected_stayingtime[:, from_state_ind]))
            for state_ind in range(0, num_states):
                transition_prob_gene[state_ind, state_ind] = -(np.sum(transition_prob_gene[state_ind, :state_ind]) + np.sum(transition_prob_gene[state_ind, state_ind+1:]))
        print('current transition rate matrix')
        print(transition_prob_gene)
        print('current transition probability matrix (delT=1)')
        print(expm(transition_prob_gene))

        if fixed_emission_prob is None: 
            # Update the parameter (emission probability)
            tmp1 = np.full((num_samples, num_states, num_subscores, num_labels), -np.inf, dtype=np.float64)
            tmp2 = np.full((num_samples, num_states, num_subscores), -np.inf, dtype=np.float64)
            for samp_ind in range(0, num_samples):
                log_gamma = log_gamma_array[samp_ind, :sequence_lengths[samp_ind]]
                for subscore_ind in range(0, num_subscores):
                    x_samp_uniq = np.unique(X[samp_ind, :sequence_lengths[samp_ind], subscore_ind])
                    for x_elem in range(0, num_labels):
                        if x_elem in x_samp_uniq:
                            tmp1[samp_ind, :, subscore_ind, x_elem] = logsumexp(log_gamma[X[samp_ind, :sequence_lengths[samp_ind], subscore_ind] == x_elem, :], axis=0)
                        else:
                            tmp1[samp_ind, :, subscore_ind, x_elem] = jnp.log(0)
                    tmp2[samp_ind, :, subscore_ind] = logsumexp(log_gamma[:, :], axis=0)
            for state_ind in range(0, num_states):
                log_emission_prob[state_ind] = logsumexp(tmp1[:, state_ind], axis=0) - np.tile(np.expand_dims(logsumexp(tmp2[:, state_ind], axis=0), axis=1), (1, num_labels))

        print('current emission prpb.')
        print('**********************************************')

        # Progression speed
        if uniform_speed is False:
            tmp_speed_log_denominator = np.full((num_samples, num_states), -np.inf, dtype=np.float64) 
            for from_state_ind in range(0, num_states):
                tmp_speed_log_denominator[:, from_state_ind] = np.log(-prev_transition_prob_gene[from_state_ind, from_state_ind]) + log_expected_stayingtime[:, from_state_ind]
            speed_log_A = logsumexp(log_expected_transition_times, axis=(1,2))
            speed_log_B = logsumexp(tmp_speed_log_denominator, axis=1)
            if for_test is True:
                res = optim.minimize(constraint_progress_speed, progress_speed, args=(speed_log_A, speed_log_B, 3), method='BFGS')
                progress_speed = res.x
            else:
                res = optim.minimize(constraint_progress_speed, progress_speed, args=(speed_log_A, speed_log_B), method='BFGS')
                progress_speed = res.x

        print("+++++++++++++")
        print('current progress speed')
        print(progress_speed)
        print("mean")
        print(np.mean(progress_speed))
        print("+++++++++++++")

    emission_prob = np.exp(log_emission_prob)
    initial_z_prob = np.exp(log_initial_z_prob)

    ###### Viterbi algorithm for state estimation ################
    optimal_paths = []
    for samp_ind in range(num_samples):
        X_samp = X[samp_ind]
        samp_sequence_length = sequence_lengths[samp_ind]
        observation_timing = T[samp_ind, 0:samp_sequence_length]
        deltaT = T[samp_ind, 1:samp_sequence_length] - T[samp_ind, 0:samp_sequence_length - 1]
        viterbi_table = np.zeros([samp_sequence_length, num_states])
        log_transition_prob = np.zeros([samp_sequence_length-1, num_states, num_states])

        # initialize
        for state_ind in range(num_states):
            viterbi_table[0, state_ind] = log_initial_z_prob[state_ind] + np.sum(log_emission_prob[state_ind,np.arange(num_subscores), X_samp[0]])

        # recursion
        for observation_ind in range(1, samp_sequence_length):
            log_transition_prob[observation_ind-1] = np.log(expm(np.exp(progress_speed[samp_ind]) * deltaT[observation_ind-1] * transition_prob_gene))
            for state_ind in range(num_states):
                viterbi_table[observation_ind, state_ind] = np.max(viterbi_table[observation_ind - 1, :] + log_transition_prob[observation_ind - 1, :, state_ind]) + np.sum(log_emission_prob[state_ind, np.arange(num_subscores), X_samp[observation_ind]])

        # backtracking
        optimal_state = np.zeros([samp_sequence_length], dtype="int8")
        optimal_state[-1] = np.argmax(viterbi_table[-1, :])
        for observation_ind in np.arange(0, samp_sequence_length)[samp_sequence_length-2::-1]:
            optimal_state[observation_ind] = np.argmax(viterbi_table[observation_ind,:] + log_transition_prob[observation_ind,:,optimal_state[observation_ind+1]])

        optimal_paths.append(np.stack([np.squeeze(observation_timing), optimal_state]))

    return(
        initial_z_prob,
        transition_prob_gene,
        emission_prob,
        progress_speed,
        log_likelihood_trajectory,
        optimal_paths,
    )

def forward_backward_algorithm(X, log_initial_z_prob, delT, log_transition_prob, log_emission_prob):
    data_len = X.shape[0]
    num_subscores = X.shape[1]
    num_states = log_transition_prob[0].shape[0]
    delT_unique = np.unique(delT)

    # Forward algorithm
    log_alpha = np.full((num_states, data_len), -np.inf, dtype=np.float64)

    log_alpha[:, 0] = log_initial_z_prob + np.sum(log_emission_prob[:, np.arange(0, num_subscores), X[0]], axis=1)
    for z_ind in range(1, data_len):
        log_alpha[:, z_ind] = np.sum(log_emission_prob[:, np.arange(0, num_subscores), X[z_ind]], axis=1) + logsumexp(log_alpha[:, z_ind - 1] + log_transition_prob[np.where(delT_unique == delT[z_ind - 1])[0][0], :, :].transpose(),axis=1)  # logsumexp function is required for accurate computation.
    log_alpha = jnp.stack(log_alpha)

    # Backward algorithm
    log_beta = np.full((num_states, data_len), -np.inf, dtype=np.float64)
    log_beta[:, -1] = np.squeeze(np.zeros((num_states, 1)))
    for z_ind in range(2, data_len+1):
        log_beta[:, -z_ind] = logsumexp(log_beta[:, -z_ind + 1] + np.sum(log_emission_prob[:, np.arange(0, num_subscores), X[-z_ind + 1]], axis=1) + log_transition_prob[np.where(delT_unique == delT[-z_ind + 1])[0][0], :, :], axis=1)  # logsumexp function is required for accurate computation.
    log_beta = jnp.stack(log_beta)

    log_likelihood = logsumexp(log_alpha[:,-1])

    # compute single-state probability
    log_gamma = np.full((data_len, num_states), -np.inf, dtype=np.float64)
    for z_ind in range(0, data_len):
        log_gamma[z_ind, :] = log_alpha[:, z_ind] + log_beta[:, z_ind] - log_likelihood
    log_gamma = jnp.stack(log_gamma)

    # compute double-states probability
    log_xi = np.full((data_len - 1, num_states, num_states), -np.inf, dtype=np.float64)
    for z_ind in range(0, data_len - 1):
        log_xi[z_ind, :, :] = (np.tile(log_alpha[:, z_ind], (num_states, 1)).transpose() + np.tile(np.sum(log_emission_prob[:, np.arange(0, num_subscores), X[z_ind + 1]], axis=1),(num_states, 1)) + log_transition_prob[np.where(delT_unique == delT[z_ind])[0][0], :, :] + np.tile(log_beta[:, z_ind + 1], (num_states, 1))) - log_likelihood
    log_xi = jnp.stack(log_xi)

    return(
        log_likelihood,
        log_gamma,
        log_xi,
    )


def compute_counttable(num_states, delT, log_xi):
    delT_unique = jnp.unique(delT)
    num_delT_unique = delT_unique.shape[0]

    log_count_table = np.full((num_delT_unique, num_states, num_states), -np.inf, dtype=np.float64) 
    for dt_ind, dt in enumerate(delT_unique):
        t_inds = jnp.where(delT == dt)
        t_inds = np.stack(t_inds)[0]
        log_count_table[dt_ind, :, :] = logsumexp(log_xi[t_inds, :, :], axis=0)

    return log_count_table


def expectation_expm_method(num_states, delT_unique, progress_speed, transition_prob_gene, log_transition_prob, log_count_table):
    num_delT_unique = delT_unique.shape[0]

    # Compute expected staying time at state i
    log_expected_staying_time = np.full((num_states), -np.inf, dtype=np.float64)
    log_expected_staying_time_dt = np.full((num_states,num_delT_unique), -np.inf, dtype=np.float64) 
    for state_ind in range(0, num_states):
        for dt_ind, dt in enumerate(delT_unique):
            log_D = np.zeros((num_states, num_states))
            A = np.zeros((2 * num_states, 2 * num_states))   
            A[:num_states, :num_states] = np.exp(progress_speed) * transition_prob_gene
            A[state_ind, num_states + state_ind] = 1
            A[num_states:, num_states:] = np.exp(progress_speed) * transition_prob_gene
            expA = expm(dt * A)
            part_expA = np.stack(expA[:num_states, num_states:])
            log_D_tmp = jnp.log(part_expA) - log_transition_prob[dt_ind]
            log_D = log_D_tmp
            tmp = log_count_table[dt_ind, :, :] + log_D
            tmp = tmp.at[np.isnan(tmp)].set(0)
            log_expected_staying_time_dt[state_ind, dt_ind] = logsumexp(np.triu(tmp)[np.nonzero(np.triu(tmp))]) #to modify: compute only Non-Nan
        log_expected_staying_time[state_ind] = logsumexp(log_expected_staying_time_dt[state_ind, :])

    # Compute expected times of transition from state i to state j
    log_expected_transition_times = np.full((num_states,num_states), -np.inf, dtype=np.float64)    
    log_expected_transition_times_dt = np.full((num_states, num_states, num_delT_unique), -np.inf, dtype=np.float64) 
    for from_state_ind in range(0, num_states-1):
        for to_state_ind in range(from_state_ind, num_states):
            if from_state_ind != to_state_ind:
                for dt_ind, dt in enumerate(delT_unique):
                    A = np.zeros([2 * num_states, 2 * num_states])
                    A[:num_states, :num_states] = np.exp(progress_speed) * transition_prob_gene
                    A[from_state_ind, num_states + to_state_ind] = 1
                    A[num_states:, num_states:] = np.exp(progress_speed) * transition_prob_gene
                    expA = expm(dt * A)
                    part_expA = np.stack(expA[:num_states, num_states:])
                    log_N_tmp = progress_speed + jnp.log(transition_prob_gene[from_state_ind,to_state_ind]) + jnp.log(part_expA) - log_transition_prob[dt_ind]
                    log_N = log_N_tmp
                    tmp = log_count_table[dt_ind, :, :] + log_N
                    tmp = tmp.at[np.isnan(tmp)].set(0)
                    log_expected_transition_times_dt[from_state_ind, to_state_ind, dt_ind] = logsumexp(np.triu(tmp)[np.nonzero(np.triu(tmp))])
                log_expected_transition_times[from_state_ind, to_state_ind] = logsumexp(log_expected_transition_times_dt[from_state_ind, to_state_ind, :])

    return (
        log_expected_staying_time,
        log_expected_transition_times,
    )


def constraint_progress_speed(progress_speed, log_A, log_B, sigma = 5):
    tmp = np.exp(log_A) - np.exp(log_B) * np.exp(progress_speed) - (1/sigma**2) * progress_speed
    return np.sum(np.sqrt(tmp ** 2))



def cthmm_likelihood(num_states, num_samples, X, T, sequence_lengths, initial_z_prob, transition_prob_gene, emission_prob, progress_speed):
    print("--------------cthmm---------------")

    deltaT_unique_tmp = np.array([])
    for samp_ind in range(0, num_samples):
        deltaT = T[samp_ind, 1:sequence_lengths[samp_ind]] - T[samp_ind, 0:sequence_lengths[samp_ind] - 1]
        deltaT_unique_tmp = np.concatenate((deltaT_unique_tmp, np.unique(deltaT)))
    print(deltaT_unique_tmp)
    total_deltaT_unique = np.asarray(np.unique(deltaT_unique_tmp), dtype=np.int16)
    num_total_deltaT_unique = total_deltaT_unique.shape[0]
    print(total_deltaT_unique)

    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("Compute likelihood: ")
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    log_gamma_array = np.full((num_samples, np.max(sequence_lengths), num_states), -np.inf, dtype=np.float64) 
    log_xi_array = np.full((num_samples, np.max(sequence_lengths) - 1, num_states, num_states), -np.inf, dtype=np.float64) 

    log_likelihood_array = np.full((num_samples), -np.inf, dtype=np.float64)
    for samp_ind in range(0, num_samples):
        deltaT = T[samp_ind, 1:sequence_lengths[samp_ind]] - T[samp_ind, 0:sequence_lengths[samp_ind]-1]
        deltaT_unique = np.unique(deltaT)
        num_deltaT_unique = deltaT_unique.shape[0]

        transition_prob = np.ones([num_deltaT_unique, num_states, num_states])
        for delT_ind, delT in enumerate(deltaT_unique):
            transition_prob[delT_ind] = expm(np.exp(progress_speed[samp_ind]) * delT * transition_prob_gene)
        transition_prob = jnp.stack(transition_prob)
        log_transition_prob = np.log(transition_prob)

        log_initial_z_prob = np.log(initial_z_prob)
        log_emission_prob = np.log(emission_prob)

        X_samp = X[samp_ind, :sequence_lengths[samp_ind]]

        (log_likelihood_tmp, log_gamma_array[samp_ind, 0:sequence_lengths[samp_ind], :], log_xi_array[samp_ind, 0:sequence_lengths[samp_ind]-1, :, :]) = forward_backward_algorithm(X_samp, log_initial_z_prob, deltaT, log_transition_prob, log_emission_prob)
        log_likelihood_array[samp_ind] = log_likelihood_tmp
    log_likelihood = np.sum(log_likelihood_array)
    print('## log-likelihood ##')
    print(log_likelihood)
    print('####')

    return(
        log_likelihood,
    )
