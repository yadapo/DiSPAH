#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing


def ALSdataprep():
    subject_metadata_df = pd.read_csv('Path to AnswerALS metadata')
    subject_covar_df = pd.read_csv('Path to AnswerALS sub data')
    ALSFRSR_df = pd.read_csv('Path to AnswerALS ALSFRS_R data')
    medication_df = pd.read_csv('Path to AnswerALS medication data')


    ALSsubject_metadata_df = subject_metadata_df[subject_metadata_df['Cohort'] == 'ALS']
    num_ALSsubject = ALSsubject_metadata_df.shape[0]
    tmp = np.zeros(num_ALSsubject)
    OnsetSite = []
    for ALSsubj_ind in range(num_ALSsubject):
        tmp[ALSsubj_ind] =  len(np.where(ALSFRSR_df['SubjectUID'] == ALSsubject_metadata_df['GUID'].iloc[ALSsubj_ind])[0])
        OnsetSite.append(subject_covar_df['Site of Onset'][subject_covar_df['GUID'] == ALSsubject_metadata_df['GUID'].iloc[ALSsubj_ind]].values[0])        

    ALSsubject_metadata_df['ALSFRSR_Visit_Times'] = tmp
    ALSsubject_metadata_df['Site of Onset'] = OnsetSite
    min_visit_times = 4
    ALSusesubject_metadata_df = ALSsubject_metadata_df[ALSsubject_metadata_df['ALSFRSR_Visit_Times'] >= min_visit_times]
    ALSusesubject_metadata_df = ALSusesubject_metadata_df[ALSusesubject_metadata_df['Site of Onset'] == 'Limb']

    ALSFRSR_df['alsfrs5'] = ALSFRSR_df['alsfrs5a'].fillna(0) + ALSFRSR_df['alsfrs5b'].fillna(0)
    use_subjectID = ALSusesubject_metadata_df["GUID"].to_list()
    ALSusesubject_covar_df = subject_covar_df[subject_covar_df['GUID'].isin(use_subjectID)]
    ALSusesubject_covar_df = ALSusesubject_covar_df.set_index('GUID').reindex(use_subjectID).reset_index()
    max_visit_time = np.int8(np.max(ALSusesubject_covar_df['Number of Visits']))
    score_num = 12
    ALSFRSR_total = []
    ALSFRSRtime_np = np.zeros((len(use_subjectID), max_visit_time))
    ALSFRSRTscore_np = np.zeros((len(use_subjectID), max_visit_time))
    ALSFRSRscore_np = np.zeros((len(use_subjectID), max_visit_time, score_num))
    ALSFRSRsequence_length_np = np.zeros(len(use_subjectID), dtype=np.int8)
    ALSFRSRdt_zero_samp = np.zeros(len(use_subjectID))
    for subj_ind in range(len(use_subjectID)):
        tmp_time_day_np = ALSFRSR_df[ALSFRSR_df['SubjectUID'] == ALSusesubject_metadata_df["GUID"].iloc[subj_ind]]['Visit_Date'].values
        tmp_time_np = np.floor(tmp_time_day_np / 7) # Convert day to week.
        tmp_alsfrs_np = ALSFRSR_df[ALSFRSR_df['SubjectUID'] == ALSusesubject_metadata_df["GUID"].iloc[subj_ind]][['alsfrs1','alsfrs2','alsfrs3','alsfrs4','alsfrs5','alsfrs6','alsfrs7','alsfrs8','alsfrs9','alsfrsr1','alsfrsr2','alsfrsr3']].values
        tmp_alsfrs_np = tmp_alsfrs_np[np.argsort(tmp_time_np)]
        tmp_alsfrst_np =  np.sum(tmp_alsfrs_np, axis=1)
        tmp_time_np = np.sort(tmp_time_np)
        tmp_time_np = np.delete(tmp_time_np, np.where(np.isnan(np.sum(tmp_alsfrs_np,axis=1))))
        tmp_alsfrst_np = np.delete(tmp_alsfrst_np, np.where(np.isnan(np.sum(tmp_alsfrs_np,axis=1))))
        tmp_alsfrs_np = np.delete(tmp_alsfrs_np, np.where(np.isnan(np.sum(tmp_alsfrs_np,axis=1))), axis=0)
        ALSFRSRsequence_length_np[subj_ind] = tmp_alsfrst_np.shape[0]
        ALSFRSR_total.append([tmp_time_np.tolist(), tmp_alsfrst_np.tolist()])
        ALSFRSRtime_np[subj_ind, :tmp_time_np.shape[0]] = tmp_time_np
        ALSFRSRTscore_np[subj_ind, :tmp_time_np.shape[0]] = tmp_alsfrst_np
        ALSFRSRscore_np[subj_ind, :tmp_time_np.shape[0], :] = tmp_alsfrs_np
        ALSFRSRdt_zero_samp[subj_ind] = np.sum((tmp_time_np[1:] - tmp_time_np[:-1]) == 0)

    ALSFRSRtime_np = ALSFRSRtime_np[np.where(1-ALSFRSRdt_zero_samp)]
    ALSFRSRTscore_np = ALSFRSRTscore_np[np.where(1 - ALSFRSRdt_zero_samp)]
    ALSFRSRscore_np = ALSFRSRscore_np[np.where(1 - ALSFRSRdt_zero_samp)]
    ALSFRSRsequence_length_np = ALSFRSRsequence_length_np[np.where(1-ALSFRSRdt_zero_samp)]
    ALSusesubject_covar_df = ALSusesubject_covar_df.iloc[np.where(1 - ALSFRSRdt_zero_samp)[0], :]
    ALSusesubject_metadata_df = ALSusesubject_metadata_df.iloc[np.where(1-ALSFRSRdt_zero_samp)[0],:]
    ALSFRSRtime_np =  ALSFRSRtime_np[ALSFRSRsequence_length_np>3,:]
    ALSFRSRTscore_np = ALSFRSRTscore_np[ALSFRSRsequence_length_np > 3, :]
    ALSFRSRscore_np = ALSFRSRscore_np[ALSFRSRsequence_length_np > 3, :, :]
    ALSusesubject_metadata_df = ALSusesubject_metadata_df.iloc[ALSFRSRsequence_length_np > 3, :]
    ALSusesubject_covar_df = ALSusesubject_covar_df.iloc[ALSFRSRsequence_length_np > 3, :]
    ALSFRSRsequence_length_np = ALSFRSRsequence_length_np[ALSFRSRsequence_length_np>3]
    ALSFRSRtime_np = ALSFRSRtime_np[np.sum(ALSFRSRscore_np[:,0],axis=1) > 20, :]
    ALSFRSRsequence_length_np = ALSFRSRsequence_length_np[np.sum(ALSFRSRscore_np[:,0],axis=1) > 20]
    ALSusesubject_metadata_df = ALSusesubject_metadata_df.iloc[np.sum(ALSFRSRscore_np[:,0],axis=1) > 20]
    ALSusesubject_covar_df = ALSusesubject_covar_df.iloc[np.sum(ALSFRSRscore_np[:,0],axis=1) > 20]
    ALSFRSRTscore_np = ALSFRSRTscore_np[np.sum(ALSFRSRscore_np[:,0],axis=1) > 20, :]
    ALSFRSRscore_np = ALSFRSRscore_np[np.sum(ALSFRSRscore_np[:,0],axis=1) > 20, :, :]
    ALSFRSRT_np = np.stack([ALSFRSRtime_np, ALSFRSRTscore_np], axis=1)
    ALSFRSRtime_np = np.expand_dims(ALSFRSRtime_np, axis=2)
    ALSFRSR_np = np.concatenate([ALSFRSRtime_np, ALSFRSRscore_np], axis=2)
    use_subjectID = ALSusesubject_metadata_df['GUID']

    plt.figure(figsize=[8, 4])
    subj_cnt = 0
    ALSFRS_selec_total = []
    for subj_ind in range(len(use_subjectID)):
        plt.plot(ALSFRSRtime_np[subj_ind,:ALSFRSRsequence_length_np[subj_ind]],ALSFRSRTscore_np[subj_ind,:ALSFRSRsequence_length_np[subj_ind]], 'ko-', linewidth=0.5, alpha=0.4, markersize=3)
        subj_cnt += 1
        ALSFRS_selec_total.append(ALSFRSR_total[subj_ind])
    plt.xlim(0, 200)
    plt.ylim(0, 50)
    plt.xlabel('Weeks from the first visit')
    plt.ylabel('Total ALSFRS score')
    plt.savefig('ALSFRStotalscore.svg', dpi=300)
    plt.savefig('ALSFRStotalscore.jpg', dpi=300)
    plt.show()
    print(subj_cnt)

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=None, top=None, wspace=0.3, hspace=0.3)
    for subj_ind in range(len(use_subjectID)):
        for subscore_ind in range(score_num):
            plot_row = np.int16(subscore_ind % 3)
            plot_col = np.int16(np.floor(subscore_ind / 3))
            axes[plot_row][plot_col].plot(ALSFRSRtime_np[subj_ind, :np.int16(ALSFRSRsequence_length_np[subj_ind])],
                     ALSFRSRscore_np[subj_ind, :np.int16(ALSFRSRsequence_length_np[subj_ind]), subscore_ind], 'ko-', linewidth=0.5,
                     alpha=0.4, markersize=3)
            axes[plot_row][plot_col].set_xlim((0, 200))
            axes[plot_row][plot_col].set_ylim((0, 4))
            axes[plot_row][plot_col].set_xticks([0, 100, 200])
            axes[plot_row][plot_col].set_xticklabels([0, 100, 200], fontsize=24)
            axes[plot_row][plot_col].set_yticks([0, 1, 2, 3, 4])
            axes[plot_row][plot_col].set_yticklabels([0, 1, 2, 3, 4], fontsize=24)
    # Adding common labels
    fig.text(0.5, 0.02, 'Weeks from the first visit', ha='center', va='center', fontsize=28)
    fig.text(0.06, 0.5, 'Sub-score', ha='center', va='center', rotation='vertical', fontsize=28)
    plt.savefig('ALSFRSRsubscore.svg')
    plt.savefig('ALSFRSRsubscore.jpg', dpi=300)
    plt.show()

    use_covar = ['GUID', 'Sex', 'Site of Onset', 'Age At Symptom Onset', 'ALSFRS-R Progression Slope', 'ClinReport Mutations Details', 'Has Variant (WGS)', 'C9orf72 repeat length', 'ATXN2 repeat length']
    ALSusesubject_covar_df = ALSusesubject_covar_df[use_covar]

    label_encoder = preprocessing.LabelEncoder()
    ALSusesubject_covar_df['Sex'] = label_encoder.fit_transform(ALSusesubject_covar_df['Sex'])
    ALSusesubject_covar_df['Site of Onset'] = label_encoder.fit_transform(ALSusesubject_covar_df['Site of Onset'])
    subject_mutation_array = []
    for subj_ind, subject_mutation in enumerate(ALSusesubject_covar_df['ClinReport Mutations Details'].values):
        if type(subject_mutation) == str:
            subject_mutation_array.append(subject_mutation)
        else:
            if type(ALSusesubject_covar_df['Has Variant (WGS)'].iloc[subj_ind]) == str:
                subject_mutation_array.append(
                ALSusesubject_covar_df['Has Variant (WGS)'].iloc[subj_ind].replace(" (WGS)", ""))
            else:
                subject_mutation_array.append('None')

    ALSusesubject_covar_df['Mutations'] = subject_mutation_array
    ALSusesubject_covar_df = ALSusesubject_covar_df.drop(['ClinReport Mutations Details', 'Has Variant (WGS)'], axis=1)
    ALSusesubject_covar_df.set_index('GUID')
    print(ALSusesubject_covar_df)

    ALSusesubject_covar_df['Riluzole'] = np.zeros(len(use_subjectID), dtype=bool)
    medication_df = medication_df[['SubjectUID', 'med']]
    use_subjectID = ALSusesubject_metadata_df['GUID']
    for subject_id in use_subjectID:
        tmp_subj_med_df = medication_df['med'][medication_df['SubjectUID'] == subject_id]
        if np.sum(tmp_subj_med_df.str.contains('Riluzole', case=False)):
            ALSusesubject_covar_df['Riluzole'][ALSusesubject_covar_df['GUID'] == subject_id] = True
        else:
            ALSusesubject_covar_df['Riluzole'][ALSusesubject_covar_df['GUID'] == subject_id] = False

    X = np.int8(ALSFRSRscore_np)
    obs_timings = np.int16(ALSFRSRtime_np)
    sequence_length = np.int16(ALSFRSRsequence_length_np)

    return X, obs_timings, sequence_length, ALSusesubject_metadata_df, ALSusesubject_covar_df


def ALSdataprep_PROACT():
    ALSsubject_metadata_df = pd.read_csv('Path to PROACT metadata')
    ALSsubject_onsetsite_df = pd.read_csv('Path to PROACT history data')
    ALSFRSR_df = pd.read_csv('Path to PROACT ALSFRSR data')
    riluzole_df = pd.read_csv('Path to PROACT risuzole data')
    medication_df = pd.read_csv('Path to PROACT medication data')
    medication_riluzole_df = medication_df[medication_df['Medication_Coded'].str.contains('Riluzole', case=False)]

    num_ALSsubject = ALSsubject_metadata_df.shape[0]
    tmp = np.zeros(num_ALSsubject)
    for ALSsubj_ind in range(num_ALSsubject):
        tmp[ALSsubj_ind] = len(np.where(ALSFRSR_df['subject_id'] == ALSsubject_metadata_df['subject_id'].iloc[ALSsubj_ind])[0])

    ALSsubject_metadata_df['ALSFRSR_Visit_Times'] = tmp
    ALSsubject_metadata_df = pd.merge(ALSsubject_metadata_df, ALSsubject_onsetsite_df[['subject_id', 'Site_of_Onset___Limb', 'Site_of_Onset']], on='subject_id', how='right')
    min_visit_times = 4
    max_visit_times = 20
    ALSusesubject_metadata_df = ALSsubject_metadata_df[ALSsubject_metadata_df['ALSFRSR_Visit_Times'] >= min_visit_times]
    ALSusesubject_metadata_df = ALSusesubject_metadata_df[ALSusesubject_metadata_df['ALSFRSR_Visit_Times'] <= max_visit_times]
    ALSusesubject_metadata_df = ALSusesubject_metadata_df[(ALSusesubject_metadata_df['Site_of_Onset___Limb'] == 1).values | (ALSusesubject_metadata_df['Site_of_Onset'] =='Onset: Limb').values]

    # use subjects who were evaluated with ALSFRS_R, not with old ALSFRS.
    ALSFRSR_df = ALSFRSR_df[~np.isnan(ALSFRSR_df['ALSFRS_R_Total'])]
    ALSFRSR_df['Q5'] = ALSFRSR_df['Q5a_Cutting_without_Gastrostomy'].fillna(0)
    ALSFRSR_use_subjectID = np.unique(ALSFRSR_df['subject_id'].values).tolist()
    ALSusesubject_metadata_df = ALSusesubject_metadata_df[ALSusesubject_metadata_df['subject_id'].isin(ALSFRSR_use_subjectID)]
    use_subjectID = ALSusesubject_metadata_df['subject_id'].values.tolist()

    score_num = 12
    ALSFRSR_total = []
    ALSFRSRtime_np = np.zeros((len(use_subjectID), max_visit_times))
    ALSFRSRTscore_np = np.zeros((len(use_subjectID), max_visit_times))
    ALSFRSRscore_np = np.zeros((len(use_subjectID), max_visit_times, score_num))
    ALSFRSRsequence_length_np = np.zeros(len(use_subjectID))
    ALSFRSRdt_zero_samp = np.zeros(len(use_subjectID))
    for subj_ind in range(len(use_subjectID)):
        tmp_time_np = ALSFRSR_df[ALSFRSR_df['subject_id'] == ALSusesubject_metadata_df['subject_id'].iloc[subj_ind]]['ALSFRS_Delta'].values / 7
        tmp_time_np = np.int16(tmp_time_np)# convert days to weeks
        tmp_alsfrs_np = ALSFRSR_df[ALSFRSR_df['subject_id'] == ALSusesubject_metadata_df['subject_id'].iloc[subj_ind]][['Q1_Speech','Q2_Salivation','Q3_Swallowing','Q4_Handwriting','Q5','Q6_Dressing_and_Hygiene','Q7_Turning_in_Bed','Q8_Walking','Q9_Climbing_Stairs','R_1_Dyspnea','R_2_Orthopnea','R_3_Respiratory_Insufficiency']].values
        tmp_alsfrs_np = tmp_alsfrs_np[np.argsort(tmp_time_np)]
        tmp_alsfrst_np = np.sum(tmp_alsfrs_np, axis=1)
        tmp_time_np = np.sort(tmp_time_np)
        tmp_time_np = np.delete(tmp_time_np, np.where(np.isnan(np.sum(tmp_alsfrs_np,axis=1))))
        tmp_alsfrst_np = np.delete(tmp_alsfrst_np, np.where(np.isnan(np.sum(tmp_alsfrs_np,axis=1))))
        tmp_alsfrs_np = np.delete(tmp_alsfrs_np, np.where(np.isnan(np.sum(tmp_alsfrs_np,axis=1))), axis=0)
        tmp_alsfrst_np = np.delete(tmp_alsfrst_np, np.where(np.isnan(tmp_time_np)))
        tmp_alsfrs_np = np.delete(tmp_alsfrs_np, np.where(np.isnan(tmp_time_np)), axis=0)
        tmp_time_np = np.delete(tmp_time_np, np.where(np.isnan(tmp_time_np)))
        tmp_alsfrs_np = np.delete(tmp_alsfrs_np, np.where(tmp_time_np<0), axis=0)
        tmp_alsfrst_np = np.delete(tmp_alsfrst_np, np.where(tmp_time_np<0))
        tmp_time_np = np.delete(tmp_time_np, np.where(tmp_time_np < 0))
        tmp_alsfrs_np = np.delete(tmp_alsfrs_np, np.where(tmp_time_np>200), axis=0)
        tmp_alsfrst_np = np.delete(tmp_alsfrst_np, np.where(tmp_time_np>200))
        tmp_time_np = np.delete(tmp_time_np, np.where(tmp_time_np>200))
        tmp_dt_zero_ind = np.where((tmp_time_np[1:] - tmp_time_np[:-1]) == 0)
        tmp_time_np = np.delete(tmp_time_np, tmp_dt_zero_ind)
        tmp_alsfrs_np = np.delete(tmp_alsfrs_np, tmp_dt_zero_ind, axis=0)
        tmp_alsfrst_np = np.delete(tmp_alsfrst_np, tmp_dt_zero_ind)
        ALSFRSRsequence_length_np[subj_ind] = tmp_alsfrst_np.shape[0]
        ALSFRSR_total.append([tmp_time_np.tolist(), tmp_alsfrst_np.tolist()])
        ALSFRSRtime_np[subj_ind, :tmp_time_np.shape[0]] = tmp_time_np
        ALSFRSRTscore_np[subj_ind, :tmp_time_np.shape[0]] = tmp_alsfrst_np
        ALSFRSRscore_np[subj_ind, :tmp_time_np.shape[0], :] = tmp_alsfrs_np


    ALSFRSRtime_np =  ALSFRSRtime_np[ALSFRSRsequence_length_np>3,:]
    ALSFRSRTscore_np = ALSFRSRTscore_np[ALSFRSRsequence_length_np > 3, :]
    ALSFRSRscore_np = ALSFRSRscore_np[ALSFRSRsequence_length_np > 3, :, :]
    ALSusesubject_metadata_df = ALSusesubject_metadata_df.iloc[ALSFRSRsequence_length_np > 3, :]
    ALSFRSRsequence_length_np = ALSFRSRsequence_length_np[ALSFRSRsequence_length_np>3]
    ALSFRSRtime_np = ALSFRSRtime_np[np.sum(ALSFRSRscore_np[:,0],axis=1) > 20, :]
    ALSFRSRsequence_length_np = ALSFRSRsequence_length_np[np.sum(ALSFRSRscore_np[:,0],axis=1) > 20]
    ALSusesubject_metadata_df = ALSusesubject_metadata_df.iloc[np.sum(ALSFRSRscore_np[:,0],axis=1) > 20]
    ALSFRSRTscore_np = ALSFRSRTscore_np[np.sum(ALSFRSRscore_np[:,0],axis=1) > 20, :]
    ALSFRSRscore_np = ALSFRSRscore_np[np.sum(ALSFRSRscore_np[:,0],axis=1) > 20, :, :]
    ALSFRSRT_np = np.stack([ALSFRSRtime_np, ALSFRSRTscore_np], axis=1)
    use_subjectID = ALSusesubject_metadata_df['subject_id']

    plt.figure(figsize=[8, 4])
    for subj_ind in range(len(use_subjectID)):
        plt.plot(ALSFRSRtime_np[subj_ind, :np.int16(ALSFRSRsequence_length_np[subj_ind])], ALSFRSRTscore_np[subj_ind, :np.int16(ALSFRSRsequence_length_np[subj_ind])], 'ko-', linewidth=0.5, alpha=0.4, markersize=3)
    plt.xlim(0, 200)
    plt.ylim(0, 50)
    plt.xlabel('Weeks from the first visit')
    plt.ylabel('Total ALSFRS score')
    plt.savefig('ALSFRSRtotalscore.svg', dpi=300)
    plt.savefig('ALSFRSRtotalscore.jpg', dpi=300)
    plt.show()

    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=None, top=None, wspace=0.3, hspace=0.3)
    for subj_ind in range(len(use_subjectID)):
        for subscore_ind in range(score_num):
            plot_row = np.int16(subscore_ind % 3)
            plot_col = np.int16(np.floor(subscore_ind / 3))
            axes[plot_row][plot_col].plot(ALSFRSRtime_np[subj_ind, :np.int16(ALSFRSRsequence_length_np[subj_ind])],
                     ALSFRSRscore_np[subj_ind, :np.int16(ALSFRSRsequence_length_np[subj_ind]), subscore_ind], 'ko-', linewidth=0.5,
                     alpha=0.4, markersize=3)
            axes[plot_row][plot_col].set_xlim((0, 200))
            axes[plot_row][plot_col].set_ylim((0, 4))
            axes[plot_row][plot_col].set_xticks([0, 100, 200])
            axes[plot_row][plot_col].set_xticklabels([0, 100, 200], fontsize=24)
            axes[plot_row][plot_col].set_yticks([0, 1, 2, 3, 4])
            axes[plot_row][plot_col].set_yticklabels([0, 1, 2, 3, 4], fontsize=24)
    # Adding common labels
    fig.text(0.5, 0.02, 'Time', ha='center', va='center', fontsize=28)
    fig.text(0.06, 0.5, 'Sub-score', ha='center', va='center', rotation='vertical', fontsize=28)
    plt.savefig('ALSFRSRsubscore.svg')
    plt.savefig('ALSFRSRsubscore.jpg', dpi=300)
    plt.show()

    ALSusesubject_covar_df = pd.DataFrame(columns=['subject_id', 'Sex', 'Age At Symptom Onset'])
    for subject_id in use_subjectID:
        tmp_metadata = ALSusesubject_metadata_df[ALSusesubject_metadata_df['subject_id'] == subject_id]
        tmp_onsetsitedata = ALSsubject_onsetsite_df[ALSsubject_onsetsite_df['subject_id'] == subject_id]
        if np.isnan(tmp_metadata['Date_of_Birth'].values[0]):
            baseline_age = tmp_metadata['Age'].values[0]
        else:
            baseline_age = np.abs(tmp_metadata['Date_of_Birth'].values[0] / 365)
        if ~np.isnan(tmp_onsetsitedata['Onset_Delta'].values[0]):
            onset_age = baseline_age - np.abs(tmp_onsetsitedata['Onset_Delta'].values[0] / 365)
        else:
            onset_age = None
        tmp_df = pd.DataFrame({'subject_id': [subject_id], 'Sex': tmp_metadata['Sex'].values, 'Age At Symptom Onset': [onset_age]})
        ALSusesubject_covar_df = pd.concat([ALSusesubject_covar_df, tmp_df], axis=0, ignore_index=True)

    label_encoder = preprocessing.LabelEncoder()
    ALSusesubject_covar_df['Sex'] = label_encoder.fit_transform(ALSusesubject_covar_df['Sex'])
    ALSusesubject_covar_df.set_index('subject_id')

    ALSusesubject_covar_df['Riluzole'] = np.zeros(len(use_subjectID), dtype=bool)
    use_subjectID = ALSusesubject_metadata_df['subject_id']
    for subject_id in use_subjectID:
        if np.sum(medication_riluzole_df['subject_id']==(subject_id)):
            ALSusesubject_covar_df['Riluzole'][ALSusesubject_covar_df['subject_id'] == subject_id] = True
        elif (riluzole_df[riluzole_df['subject_id']==subject_id]['Subject_used_Riluzole'] == 'Yes').values:
            ALSusesubject_covar_df['Riluzole'][ALSusesubject_covar_df['subject_id'] == subject_id] = True
        else:
            ALSusesubject_covar_df['Riluzole'][ALSusesubject_covar_df['subject_id'] == subject_id] = False

    print(ALSusesubject_covar_df)

    X = np.int8(ALSFRSRscore_np)
    obs_timings = np.int16(ALSFRSRtime_np)
    sequence_length = np.int16(ALSFRSRsequence_length_np)

    return X, obs_timings, sequence_length, ALSusesubject_metadata_df, ALSusesubject_covar_df


def ALS_transcriptome_dataprep(ALSusesubject_covar_df):
    transcriptome_df = pd.read_csv('Path to AnswerALS transcriptome matrix')
    transcriptome_df = transcriptome_df.set_index('Unnamed: 0').transpose()
    transcriptome_als_df = transcriptome_df.iloc[transcriptome_df.index.str.contains('CASE'), :]
    transcriptome_als_df.index = transcriptome_als_df.index.str.replace('CASE-', '').str.replace('-T', '').str.replace(
        '-*', '', regex=True)
    transcriptome_als_df.index = [s[:-4] for s in transcriptome_als_df.index]
    transcriptome_als_df = transcriptome_als_df.iloc[:, ~(np.asarray(list(
        map(lambda x: np.argmax(np.bincount(transcriptome_als_df.iloc[:, x])),
            np.arange(0, transcriptome_als_df.shape[1])))) == 0)]  # delete genes if the mode of the genes are zero.
    
    ## Filter out low-expression genes ##
    min_count_threshold=10
    min_samples_threshold=2
    # Keep only genes that have >= min_count_threshold in at least min_samples_threshold samples
    keep_genes = (
        (transcriptome_als_df >= min_count_threshold)
        .sum(axis=0)  # sum across samples for each gene
        >= min_samples_threshold
    )
    filtered_transcriptome_als_df = transcriptome_als_df.loc[:, keep_genes]

    ## Library size normalization (CPM) ##
    # Calculate total counts per sample
    total_counts_per_sample = filtered_transcriptome_als_df.sum(axis=1)
    # Divide each sample's gene counts by the sample's total, then multiply by 1e6 (CPM)
    cpm_transcriptome_als_df = filtered_transcriptome_als_df.div(total_counts_per_sample, axis=0) * 1e6
    
    scaler = preprocessing.StandardScaler()
    transcriptome_als_scaled_df = pd.DataFrame(scaler.fit_transform(np.log10(cpm_transcriptome_als_df + 1)),
                                               columns=cpm_transcriptome_als_df.columns, index=cpm_transcriptome_als_df.index)
    transcriptome_als_scaled_df.index.name = 'GUID'
    print(transcriptome_als_scaled_df)

    ALSusesubject_covar_transcriptome_df = ALSusesubject_covar_df.merge(transcriptome_als_scaled_df, on='GUID').set_index('GUID')

    return transcriptome_als_scaled_df, ALSusesubject_covar_transcriptome_df


def ALS_proteome_dataprep(ALSusesubject_covar_df):
    proteome_df = pd.read_csv('Path to AnswerALS proteome matrix')
    proteome_df = proteome_df.set_index('Protein').transpose()
    proteome_als_df = proteome_df.iloc[proteome_df.index.str.contains('CASE'), :]
    proteome_als_df.index = proteome_als_df.index.str.replace('CASE-', '').str.replace('-P', '').str.replace('-*', '',
                                                                                                             regex=True)
    proteome_als_df.index = [s[:-4] for s in proteome_als_df.index]
    proteome_als_df = proteome_als_df.iloc[:, ~(np.asarray(list(
        map(lambda x: np.argmax(np.bincount(proteome_als_df.iloc[:, x])),
            np.arange(0, proteome_als_df.shape[1])))) == 0)]  # delete that not express in the majority of samples.
    mask_nonzero = (proteome_als_df.values != 0)
    scaler = preprocessing.StandardScaler()
    sample_num, protein_num = proteome_als_df.shape
    protein_data = np.zeros((sample_num, protein_num))
    for protein_ind in range(protein_num):
        protein_data[mask_nonzero[:, protein_ind], protein_ind] = protein_data[mask_nonzero[:, protein_ind], protein_ind] = np.squeeze(scaler.fit_transform(np.log10(proteome_als_df.iloc[mask_nonzero[:, protein_ind], protein_ind] + 1).values.reshape(-1,1)))
    proteome_als_scaled_df = pd.DataFrame(protein_data, columns=proteome_als_df.columns, index=proteome_als_df.index)
    proteome_als_scaled_df.index.name = 'GUID'
    print(proteome_als_scaled_df)
    ALSusesubject_covar_proteome_df = ALSusesubject_covar_df.merge(proteome_als_scaled_df,
                                                                               on='GUID').set_index('GUID')

    return proteome_als_scaled_df, ALSusesubject_covar_proteome_df