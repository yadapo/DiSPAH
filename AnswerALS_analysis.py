import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, ttest_ind, probplot, norm, shapiro, f_oneway
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler, LabelEncoder
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

import mygene
import gseapy as gp

import ALSdataread

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
plt.rcParams['figure.subplot.bottom'] = 0.20
plt.rcParams['figure.subplot.left'] = 0.20


def ALS_omics_analysis(ALSusesubject_speed_covar_df, suffix=''):
    transcriptome_als_scaled_df, ALSusesubject_speed_covar_transcriptome_df = ALSdataread.ALS_transcriptome_dataprep(ALSusesubject_speed_covar_df)

    if suffix == '':
        ######
        # For cluster marker gene detection
        transcriptome_data = ALSusesubject_speed_covar_transcriptome_df.iloc[:, 14:]
        transcriptome_data_t = transcriptome_data.transpose()
        groups = ALSusesubject_speed_covar_transcriptome_df['cluster']
        # Group samples by their group
        grouped_samples = {group: transcriptome_data_t.columns[[i for i, x in enumerate(groups) if x == group]] for group in
                           set(groups)}

        # Perform ANOVA for each gene
        anova_results = {}
        for gene in transcriptome_data_t.index:
            samples = [transcriptome_data_t.loc[gene, grouped_samples[group]].values for group in grouped_samples]
            F_statistic, p_value = f_oneway(*samples)
            anova_results[gene] = p_value

        # Convert the ANOVA results to a DataFrame for easier analysis
        cluster_transcriptome_anova_df = pd.DataFrame.from_dict(anova_results, orient='index', columns=['p-value'])

        # Adjusting the p-values for multiple testing
        pvals_corrected = multipletests(cluster_transcriptome_anova_df['p-value'], method='fdr_bh', alpha=0.1)[1]
        cluster_transcriptome_anova_df['adjusted p-value'] = pvals_corrected

        cluster_transcriptome_anova_df.to_csv('cluster_transcriptome_anova.csv')

    # For the estimated progress speed #########################
    p_results = {}
    coef_results = {}
    for gene_ind, gene_name in enumerate(transcriptome_als_scaled_df.columns):
        X = sm.add_constant(
            ALSusesubject_speed_covar_transcriptome_df[['Sex', 'Age At Symptom Onset', 'Riluzole', 'C9orf72 mutation']])
        X[gene_name] = ALSusesubject_speed_covar_transcriptome_df[gene_name]
        y = ALSusesubject_speed_covar_transcriptome_df['est_progress_speed']
        model = sm.OLS(y, X).fit()
        p_results[gene_name] = model.pvalues[gene_name]
        coef_results[gene_name] = model.params[gene_name]

    genes = list(p_results.keys())
    pvalues = np.asarray([p_results[gene] for gene in genes])
    coefs = np.asarray([coef_results[gene] for gene in genes])
    gene_ranks = -np.log10(pvalues) * np.sign(coefs)
    results_df = pd.DataFrame({
        'genes': convert_ensembl_to_hgnc(genes),
        'pvalues': pvalues,
        'coefs': coefs,
        'gene_ranks': gene_ranks
    })
    results_df = results_df.sort_values(by='pvalues')
    results_df.to_csv('speed_genes_pvalues_coefs'+suffix+'.csv', index=False)
    # Correct for multiple comparisons
    rej, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh', alpha=0.05)
    corrected_results = dict(zip(genes, pvals_corrected))
    speed_transcriptome_regtest_df = pd.DataFrame.from_dict(corrected_results, orient='index', columns=['p-value'])
    speed_transcriptome_regtest_df.to_csv('speed_transcriptome_regtest'+suffix+'.csv')

    plt.figure()
    plt.plot(coefs[coefs > 0], -np.log10(pvalues[coefs > 0]), 'r.')
    plt.plot(coefs[coefs < 0], -np.log10(pvalues[coefs < 0]), 'b.')
    plt.xlabel('Coefficient')
    plt.ylabel('-log10(p-value)')
    plt.savefig('transcriptome_speed'+suffix+'.svg', dpi=300)
    plt.show()


    cutoff_level = 0.05
    gene_ranks = -np.log10(pvalues) * np.sign(coefs)
    gene_ranks_df = pd.Series(data=gene_ranks, index=convert_ensembl_to_hgnc(genes))
    gene_ranks_df = gene_ranks_df.iloc[~gene_ranks_df.index.isna()]
    # Run GSEA
    gsea_results = gp.prerank(rnk=gene_ranks_df,  # Provide the gene rankings
                              gene_sets='KEGG_2019_Human',  # Choose a gene set database
                              min_size=15,  # Minimum size of genes in the gene set to be considered
                              max_size=1000,  # Maximum size of genes in the gene set to be considered
                              outdir='gsea_kegg_output'+suffix,  # Output directory
                              no_plot=False,  # If you don't want to visualize the results
                              processes=24,  # Number of processes for parallel computation
                              permutation_num=1000,
                              seed=0)  # Seed for reproducibility
    # View results
    gsea_results.res2d.head()
    if gsea_results.res2d['FDR q-val'].iloc[0] < cutoff_level:
        ax = gp.dotplot(gsea_results.res2d, column="FDR q-val", title='KEGG_2019_Human', cmap=plt.cm.viridis, size=6,
                        figsize=(5, 9), cutoff=cutoff_level, show_ring=False, ofname='transcriptome_kegg_dotplot'+suffix+'.svg',
                        dpi=300)

    gsea_results = gp.prerank(rnk=gene_ranks_df,  # Provide the gene rankings
                              gene_sets='GO_Biological_Process_2018',  # Choose a gene set database
                              min_size=15,  # Minimum size of genes in the gene set to be considered
                              max_size=1000,  # Maximum size of genes in the gene set to be considered
                              outdir='gsea_gobp_output'+suffix,  # Output directory
                              no_plot=False,  # If you don't want to visualize the results
                              processes=24,  # Number of processes for parallel computation
                              permutation_num=1000,
                              seed=0)  # Seed for reproducibility
    # View results
    gsea_results.res2d.head()
    if gsea_results.res2d['FDR q-val'].iloc[0] < cutoff_level:
        ax = gp.dotplot(gsea_results.res2d, column="FDR q-val", title='GO_Biological_Process_2018', cmap=plt.cm.viridis,
                        size=6, figsize=(5, 9), cutoff=cutoff_level, show_ring=False,
                        ofname='transcriptome_gobp_dotplot'+suffix+'.svg', dpi=300)

    gsea_results = gp.prerank(rnk=gene_ranks_df,  # Provide the gene rankings
                              gene_sets='GO_Molecular_Function_2018',  # Choose a gene set database
                              min_size=15,  # Minimum size of genes in the gene set to be considered
                              max_size=1000,  # Maximum size of genes in the gene set to be considered
                              outdir='gsea_gomf_output'+suffix,  # Output directory
                              no_plot=False,  # If you don't want to visualize the results
                              processes=24,  # Number of processes for parallel computation
                              permutation_num=1000,
                              seed=0)  # Seed for reproducibility
    # View results
    gsea_results.res2d.head()
    if gsea_results.res2d['FDR q-val'].iloc[0] < cutoff_level:
        ax = gp.dotplot(gsea_results.res2d, column="FDR q-val", title='GO_Molecular_Function_2018', cmap=plt.cm.viridis,
                        size=6, figsize=(5, 9), cutoff=cutoff_level, show_ring=False,
                        ofname='transcriptome_gomf_dotplot'+suffix+'.svg', dpi=300)

    gsea_results = gp.prerank(rnk=gene_ranks_df,  # Provide the gene rankings
                              gene_sets='GO_Cellular_Component_2018',  # Choose a gene set database
                              min_size=15,  # Minimum size of genes in the gene set to be considered
                              max_size=1000,  # Maximum size of genes in bthe gene set to be considered
                              outdir='gsea_gocc_output'+suffix,  # Output directory
                              no_plot=False,  # If you don't want to visualize the results
                              processes=24,  # Number of processes for parallel computation
                              permutation_num=1000,
                              seed=0)  # Seed for reproducibility
    # View results
    gsea_results.res2d.head()
    if gsea_results.res2d['FDR q-val'].iloc[0] < cutoff_level:
        ax = gp.dotplot(gsea_results.res2d, column="FDR q-val", title='GO_Cellular_Component_2018', cmap=plt.cm.viridis,
                        size=6, figsize=(5, 9), cutoff=cutoff_level, show_ring=False,
                        ofname='transcriptome_gocc_dotplot'+suffix+'.svg', dpi=300)

    #---------------

    ##########################
    ## Proteome analysis
    ##########################
    proteome_als_scaled_df, ALSusesubject_speed_covar_proteome_df  = ALSdataread.ALS_proteome_dataprep(ALSusesubject_speed_covar_df)

    p_results = {}
    coef_results = {}
    for protein_ind, protein_name in enumerate(proteome_als_scaled_df.columns):
        nonzero_sample_list = np.where(ALSusesubject_speed_covar_proteome_df[protein_name] != 0)
        X = sm.add_constant(ALSusesubject_speed_covar_proteome_df[['Sex', 'Age At Symptom Onset', 'Riluzole', 'C9orf72 mutation']].iloc[nonzero_sample_list])
        X[protein_name] = ALSusesubject_speed_covar_proteome_df[protein_name].iloc[nonzero_sample_list]
        y = ALSusesubject_speed_covar_proteome_df['est_progress_speed'].iloc[nonzero_sample_list]
        model = sm.OLS(y, X).fit()
        p_results[protein_name] = model.pvalues[protein_name]
        coef_results[protein_name] = model.params[protein_name]

    proteins = list(p_results.keys())
    pvalues = np.asarray([p_results[protein] for protein in proteins])
    coefs = np.asarray([coef_results[protein] for protein in proteins])
    protein_ranks = -np.log10(pvalues) * np.sign(coefs)
    results_df = pd.DataFrame({
        'proteins': proteins,
        'pvalues': pvalues,
        'coefs': coefs,
        'gene_ranks': protein_ranks
    })
    results_df = results_df.sort_values(by='pvalues')
    results_df.to_csv('speed_proteins_pvalues_coefs'+suffix+'.csv', index=False)
    # Correct for multiple comparisons
    rej, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh', alpha=0.05)
    corrected_results = dict(zip(proteins, pvals_corrected))
    speed_proteome_regtest_df = pd.DataFrame.from_dict(corrected_results, orient='index', columns=['p-value'])
    speed_proteome_regtest_df.to_csv('speed_proteome_regtest'+suffix+'.csv')


    plt.figure()
    plt.plot(coefs[coefs > 0], -np.log10(pvalues[coefs > 0]), 'r.')
    plt.plot(coefs[coefs < 0], -np.log10(pvalues[coefs < 0]), 'b.')
    plt.xlabel('Coefficient')
    plt.ylabel('-log10(p-value)')
    plt.savefig('proteome_speed'+suffix+'.svg', dpi=300)
    plt.show()

    protein_ranks = -np.log10(pvalues) * np.sign(coefs)
    # Extract names after the "|"
    proteins = [name.split("|")[1] if "|" in name else name for name in proteins]

    protein_ranks_df = pd.Series(data=protein_ranks, index=proteins)
    # Run GSEA
    gsea_results = gp.prerank(rnk=protein_ranks_df,  # Provide the gene rankings
                              gene_sets='KEGG_2019_Human',  # Choose a gene set database
                              min_size=15,  # Minimum size of genes in the gene set to be considered
                              max_size=1000,  # Maximum size of genes in the gene set to be considered
                              outdir='gsea_protein_kegg_output'+suffix,  # Output directory
                              no_plot=False,  # If you don't want to visualize the results
                              processes=24,  # Number of processes for parallel computation
                              permutation_num=1000,
                              seed=0)  # Seed for reproducibility
    # View results
    gsea_results.res2d.head()
    if gsea_results.res2d['FDR q-val'].iloc[0] < cutoff_level:
        ax = gp.dotplot(gsea_results.res2d, column="FDR q-val", title='KEGG_2019_Human/protein', cmap=plt.cm.viridis, size=6,
                        figsize=(5, 9), cutoff=cutoff_level, show_ring=False, ofname='proteome_kegg_dotplot'+suffix+'.svg',
                        dpi=300)

    gsea_results = gp.prerank(rnk=protein_ranks_df,  # Provide the gene rankings
                              gene_sets='GO_Biological_Process_2018',  # Choose a gene set database
                              min_size=15,  # Minimum size of genes in the gene set to be considered
                              max_size=1000,  # Maximum size of genes in the gene set to be considered
                              outdir='gsea_protein_gobp_output'+suffix,  # Output directory
                              no_plot=False,  # If you don't want to visualize the results
                              processes=24,  # Number of processes for parallel computation
                              permutation_num=1000,
                              seed=0)  # Seed for reproducibility
    # View results
    gsea_results.res2d.head()
    if gsea_results.res2d['FDR q-val'].iloc[0] < cutoff_level:
        ax = gp.dotplot(gsea_results.res2d, column="FDR q-val", title='GO_Biological_Process_2018 protein',
                        cmap=plt.cm.viridis,
                        size=6, figsize=(5, 9), cutoff=cutoff_level, show_ring=False,
                        ofname='proteome_gobp_dotplot'+suffix+'.svg', dpi=300)

    gsea_results = gp.prerank(rnk=protein_ranks_df,  # Provide the gene rankings
                              gene_sets='GO_Molecular_Function_2018',  # Choose a gene set database
                              min_size=15,  # Minimum size of genes in the gene set to be considered
                              max_size=1000,  # Maximum size of genes in the gene set to be considered
                              outdir='gsea_protein_gomf_output'+suffix,  # Output directory
                              no_plot=False,  # If you don't want to visualize the results
                              processes=24,  # Number of processes for parallel computation
                              permutation_num=1000,
                              seed=0)  # Seed for reproducibility
    # View results
    gsea_results.res2d.head()
    if gsea_results.res2d['FDR q-val'].iloc[0] < cutoff_level:
        ax = gp.dotplot(gsea_results.res2d, column="FDR q-val", title='GO_Molecular_Function_2018', cmap=plt.cm.viridis,
                        size=6, figsize=(5, 9), cutoff=cutoff_level, show_ring=False,
                        ofname='protetome_gomf_dotplot'+suffix+'.svg', dpi=300)

    gsea_results = gp.prerank(rnk=protein_ranks_df,  # Provide the gene rankings
                              gene_sets='GO_Cellular_Component_2018',  # Choose a gene set database
                              min_size=15,  # Minimum size of genes in the gene set to be considered
                              max_size=1000,  # Maximum size of genes in the gene set to be considered
                              outdir='gsea_protein_gocc_output'+suffix,  # Output directory
                              no_plot=False,  # If you don't want to visualize the results
                              processes=24,  # Number of processes for parallel computation
                              permutation_num=1000,
                              seed=0)  # Seed for reproducibility
    # View results
    gsea_results.res2d.head()
    if gsea_results.res2d['FDR q-val'].iloc[0] < cutoff_level:
        ax = gp.dotplot(gsea_results.res2d, column="FDR q-val", title='GO_Cellular_Component_2018', cmap=plt.cm.viridis,
                        size=6, figsize=(5, 9), cutoff=cutoff_level, show_ring=False,
                        ofname='proteome_gocc_dotplot'+suffix+'.svg', dpi=300)

    return ALSusesubject_speed_covar_transcriptome_df  # , ALSusesubject_speed_covar_proteome_df


def benjamini_hochberg(p_values, Q=0.05):
    """
    Perform the Benjamini-Hochberg procedure to adjust p-values for multiple comparisons.

    Parameters:
    - p_values (list or array): Array of p-values to be adjusted.
    - Q (float): Desired false discovery rate.

    Returns:
    - List of adjusted p-values.
    """
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p_values = np.array(p_values)[sorted_indices]

    # Calculate BH critical values
    bh_critical_values = [(i / m) * Q for i in range(1, m + 1)]

    # Find the largest p-value that is less than its BH critical value
    significant = np.where(sorted_p_values <= bh_critical_values)[0]
    if len(significant) == 0:
        return [1] * m
    else:
        threshold = sorted_p_values[max(significant)]
        return [p if p <= threshold else 1 for p in p_values]


def convert_ensembl_to_hgnc(ensembl_ids):
    """
    Convert a list of Ensembl IDs to HGNC symbols.

    Parameters:
    - ensembl_ids: List of Ensembl gene IDs.

    Returns:
    - Dictionary with Ensembl IDs as keys and HGNC symbols as values.
    """
    mg = mygene.MyGeneInfo()
    result = mg.querymany(ensembl_ids, scopes='ensembl.gene', fields='symbol', species='human')

    # Create a dictionary to store the mapping
    symbol = []
    pre_ens_id = 'None'
    for ind, entry in enumerate(result):
        hgnc_symbol = entry.get('symbol', None)
        ens_id = entry.get('query', None)
        if ens_id != pre_ens_id:
            symbol.append(hgnc_symbol)
        pre_ens_id = ens_id
    return symbol

def line_fit(x, alpha, beta, gamma, c):
    sex, age, drug = x
    return alpha * sex + beta * age + gamma * drug + c

def main():
    ##############################################
    est_param_path = './result1/'
    ##############################################r
    AnswerALSdata_results = pd.read_csv(est_param_path + 'AnswerALS_covar_estimated_results.csv')

    ###################################################################################################################
    AnswerALSdata_results = AnswerALSdata_results.dropna(subset=['est_progress_speed', 'Age At Symptom Onset'])
    ####

    plt.figure(figsize=(5, 5))
    sns.swarmplot(x='Sex',y='est_progress_speed',data=AnswerALSdata_results)
    plt.ylabel('Estimated progression speed')
    plt.xticks([0, 1],['Female','Male'])
    plt.savefig('AnswerALS_est_progress_speed_sex.svg', dpi=300)
    plt.show()
    female_speed = AnswerALSdata_results['est_progress_speed'][AnswerALSdata_results['Sex']==0]
    male_speed = AnswerALSdata_results['est_progress_speed'][AnswerALSdata_results['Sex']==1]
    t_stat, p_value = ttest_ind(female_speed, male_speed)
    print("sex difference")
    print("t-statistic:", t_stat)
    print("p-value:", p_value)

    plt.figure(figsize=(5, 5))
    plt.scatter(AnswerALSdata_results['Age At Symptom Onset'], AnswerALSdata_results['est_progress_speed'])
    plt.xlabel('Age at symptom onset')
    plt.ylabel('Exp(Estimated progression speed)')
    plt.savefig('AnswerALS_est_progress_speed_onset_age.svg', dpi=300)
    plt.show()
    age_est_corr, age_est_corr_p = pearsonr(AnswerALSdata_results['Age At Symptom Onset'], AnswerALSdata_results['est_progress_speed'])
    print('correlation between age of onset and estimated progress speed')
    print('corr: ' + str(age_est_corr))
    print('p-value: ' + str(age_est_corr_p))

    plt.figure(figsize=(5, 5))
    sns.swarmplot(x='Riluzole',y='est_progress_speed',data=AnswerALSdata_results)
    plt.ylabel('Estimated progression speed')
    plt.xticks([0, 1],['Naive','Riluzole'])
    plt.savefig('AnswerALS_est_progress_speed_riluzole.svg', dpi=300)
    plt.show()
    naive_speed = AnswerALSdata_results['est_progress_speed'][AnswerALSdata_results['Riluzole']==0]
    riluzole_speed = AnswerALSdata_results['est_progress_speed'][AnswerALSdata_results['Riluzole']==1]
    t_stat, p_value = ttest_ind(naive_speed, riluzole_speed)
    print("riluzole treatment difference")
    print("t-statistic:", t_stat)
    print("p-value:", p_value)
    AnswerALSdata_results['Riluzole'] = np.int8(AnswerALSdata_results['Riluzole'])

    print('############################################')
    subject_covar_df = pd.read_csv('/data1/AnswerALS/metadata/aals_dataportal_datatable_04182023.csv')
    AnswerALSdata_results = pd.merge(AnswerALSdata_results,subject_covar_df[['GUID','Site of Onset']], on='GUID', how='inner')

    ######
    colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    color_dict = {1:'C1',2:'C2', 3:'C3', 4:'C4', 5:'C5', 6:'C6'}
    # Estimated speed for each site of onset groups
    plt.figure(figsize=(5, 5))
    sns.swarmplot(x='cluster', y='est_progress_speed', data=AnswerALSdata_results, palette=colors)
    plt.xlabel('Cluster#')
    plt.ylabel('Estimated progression speed')
    plt.xticks(['1','2','3','4','5','6'])
    plt.ylim([-5.2,5.2])
    plt.savefig('AnswerALS_cluster_est_progress_speed.jpg', dpi=300)
    plt.savefig('AnswerALS_cluster_est_progress_speed.svg', dpi=300)
    plt.show()
    
    AnswerALSdata_results['Sex'][AnswerALSdata_results['Sex']==0] = -1
    scaler = StandardScaler()
    AnswerALSdata_results['Age At Symptom Onset'] = scaler.fit_transform(AnswerALSdata_results['Age At Symptom Onset'].values.reshape(-1,1))
    print(AnswerALSdata_results['Age At Symptom Onset'] )
    popt, pcov = curve_fit(line_fit, (AnswerALSdata_results['Sex'], AnswerALSdata_results['Age At Symptom Onset'], AnswerALSdata_results['Riluzole']), AnswerALSdata_results['est_progress_speed'])
    print(popt)
    AnswerALSdata_results['residual'] = AnswerALSdata_results['est_progress_speed'] - (popt[0]*AnswerALSdata_results['Sex'] + popt[1]*AnswerALSdata_results['Age At Symptom Onset'] + popt[2]*AnswerALSdata_results['Riluzole'] + popt[3])

    popt, pcov = curve_fit(line_fit, (AnswerALSdata_results['Sex'], AnswerALSdata_results['Age At Symptom Onset'], AnswerALSdata_results['Riluzole']), AnswerALSdata_results['ALSFRS-R Progression Slope'])
    print(popt)
    AnswerALSdata_results['ALSFRS-R Progression Slope'] = AnswerALSdata_results['ALSFRS-R Progression Slope'] - AnswerALSdata_results['est_progress_speed'] - (popt[0]*AnswerALSdata_results['Sex'] + popt[1]*AnswerALSdata_results['Age At Symptom Onset'] + popt[2]*AnswerALSdata_results['Riluzole'] + popt[3])

    AnswerALSdata_results['C9orf72 mutation'] = np.int8((AnswerALSdata_results['C9orf72 repeat length']) > 100) | np.int8(AnswerALSdata_results['Mutations']=='C9orf72')
    X = sm.add_constant(AnswerALSdata_results[['Sex', 'Age At Symptom Onset', 'Riluzole']])
    X['C9orf72 mutation'] = AnswerALSdata_results['C9orf72 mutation']
    y = AnswerALSdata_results['est_progress_speed']
    model = sm.OLS(y, X).fit()
    print(model.summary())

    AnswerALSdata_results['ATXN2 mutation'] = np.int8(AnswerALSdata_results['ATXN2 repeat length'] >= 27) | np.int8(AnswerALSdata_results['Mutations']=='ATXN2')

    ######
    # Show patients who have mutations in addition to c9orf72
    print('########## Additional mutations the patients who have C9orf72 have #########')
    print(AnswerALSdata_results['Mutations'][AnswerALSdata_results['C9orf72 mutation'] == 1])
    print('##')
    print(AnswerALSdata_results['C9orf72 mutation'][AnswerALSdata_results['Mutations'] == 'C9orf72'])
    print('#################################################################################')
    ###############################################

    ######
    # Show patients who have mutations in addition to c9orf72
    print('########## Additional mutations the patients who have ATXN2 have #########')
    print(AnswerALSdata_results['Mutations'][AnswerALSdata_results['ATXN2 mutation'] == 1])
    print('#################################################################################')
    ###############################################

    print('Number of C9orf72 and ATXN2 mutations')
    print('C9orf72:')
    print(len(np.where(AnswerALSdata_results['C9orf72 mutation'] == 1)[0]))
    print(len(np.where(AnswerALSdata_results['Mutations'] == 'C9orf72')[0]))
    print('ATXN2:')
    print(len(np.where(AnswerALSdata_results['ATXN2 mutation'] == 1)[0]))
    print(len(np.where(AnswerALSdata_results['Mutations'] == 'ATXN2')[0]))
    ####
    
    ##### Duplicate the patient who have both C9orf72 mutation and ATXN2 mutation ##########
    # Select the row you want to duplicate, e.g., the first row (index 0)
    row_to_duplicate = np.where((AnswerALSdata_results['C9orf72 mutation'] == 1) & (AnswerALSdata_results['ATXN2 mutation'] == 1))[0]
    print(row_to_duplicate)
    # Append the duplicated row back to the DataFrame
    AnswerALSdata_results_forplot = AnswerALSdata_results.copy()
    print(AnswerALSdata_results_forplot)
    AnswerALSdata_results_forplot['Mutations'][AnswerALSdata_results_forplot['ATXN2 mutation'] == 1] = 'ATXN2'
    print(AnswerALSdata_results_forplot[AnswerALSdata_results_forplot['Mutations'] == 'ATXN2'])
    AnswerALSdata_results_forplot = pd.concat([AnswerALSdata_results_forplot,AnswerALSdata_results.copy().iloc[row_to_duplicate]], ignore_index=True)
    AnswerALSdata_results_forplot['Mutations'][AnswerALSdata_results_forplot['C9orf72 mutation'] == 1] = 'C9orf72'      
    print(AnswerALSdata_results_forplot[AnswerALSdata_results_forplot['Mutations'] == 'ATXN2'])
    AnswerALSdata_results_forplot['Mutations'].iloc[row_to_duplicate] = 'ATXN2'
    print(AnswerALSdata_results_forplot['Mutations'])
    print('C9orf72:')
    print(len(np.where(AnswerALSdata_results_forplot['Mutations'] == 'C9orf72')[0]))
    print('ATXN2:')
    print(len(np.where(AnswerALSdata_results_forplot['Mutations'] == 'ATXN2')[0]))
    plt.figure(figsize=(9, 5))
    AnswerALSdata_results_forplot['Mutations'][AnswerALSdata_results_forplot['Mutations'].isna()] = 'No data'
    sns.swarmplot(x='Mutations',y='est_progress_speed',data=AnswerALSdata_results_forplot)
    plt.ylabel('Estimated progression speed')
    plt.savefig('AnswerALS_mutation_est_progress_speed.jpg', dpi=300)
    plt.savefig('AnswerALS_mutation_est_progress_speed.svg', dpi=300)
    plt.show()


    print('--- C9orf72 regression ---')
    print(X)
    predicted_values = model.predict(X)
    residuals = AnswerALSdata_results['est_progress_speed'] - predicted_values
    plt.figure(figsize=(5, 5))
    plt.scatter(predicted_values, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values')
    plt.savefig("c9orf72_speed_residuals.svg", dpi=300)
    plt.show()

    # Q-Q plot
    plt.figure(figsize=(5, 5))
    probplot(residuals, plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.savefig("c9orf72_speed_residualsQQ.svg", dpi=300)
    plt.show()

    # Perform the Shapiro-Wilk test
    stat, p_value = shapiro(residuals)

    print('Statistics=%.3f, p=%.3f' % (stat, p_value))

    # Interpret the result
    alpha = 0.05
    if p_value > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')


    coefficients = model.params
    conf_int = model.conf_int(alpha=0.05)
    # Plotting
    plt.rcParams['figure.subplot.bottom'] = 0.40
    plt.rcParams['figure.subplot.left'] = 0.18
    plt.figure(figsize=(4.8, 7))
    plt.errorbar(coefficients.index, coefficients.values, yerr=[coefficients.values - conf_int[0], conf_int[1] - coefficients.values], fmt='o')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(rotation=60)
    plt.ylabel("Coefficient")
    plt.savefig("c9orf72_speed_confit.jpg", dpi=300)
    plt.savefig("c9orf72_speed_confit.svg", dpi=300)
    plt.show()
    plt.rcParams['figure.subplot.bottom'] = 0.20
    plt.rcParams['figure.subplot.left'] = 0.20
    #-------------------------------------------------------------------------------------
    
    #-------------------------------------------------------------------------------------
    # Create a scatter plot with regression lines
    X = sm.add_constant(AnswerALSdata_results[['Sex', 'Age At Symptom Onset', 'Riluzole']])
    X['ATXN2 mutation'] = np.int8(AnswerALSdata_results['ATXN2 mutation'])
    y = AnswerALSdata_results['est_progress_speed']
    model = sm.OLS(y, X).fit()
    print(model.summary())

    print('--- ATXN2 regression ---')
    print(X)
    predicted_values = model.predict(X)
    residuals = AnswerALSdata_results['est_progress_speed'] - predicted_values
    plt.figure(figsize=(5, 5))
    plt.scatter(predicted_values, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values')
    plt.savefig("ATXN2_speed_residuals.svg", dpi=300)
    plt.show()

    # Q-Q plot
    plt.figure(figsize=(5, 5))
    probplot(residuals, plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.savefig("ATXN2_speed_residualsQQ.svg", dpi=300)
    plt.show()

    coefficients = model.params
    conf_int = model.conf_int(alpha=0.05)
    # Plotting
    plt.rcParams['figure.subplot.bottom'] = 0.40
    plt.rcParams['figure.subplot.left'] = 0.18
    plt.figure(figsize=(4.8, 7))
    plt.errorbar(coefficients.index, coefficients.values, yerr=[coefficients.values - conf_int[0], conf_int[1] - coefficients.values], fmt='o')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(rotation=60)
    plt.ylabel("Coefficient")
    plt.savefig("ATXN2_speed_confit.svg", dpi=300)
    plt.show()
    plt.rcParams['figure.subplot.bottom'] = 0.20
    plt.rcParams['figure.subplot.left'] = 0.20
    #-------------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------------
    AnswerALSdata_results['SOD1 mutation'] = np.int8(AnswerALSdata_results['Mutations']=='SOD1')
    # Create a scatter plot with regression lines
    X = sm.add_constant(AnswerALSdata_results[['Sex', 'Age At Symptom Onset', 'Riluzole']])
    X['SOD1 mutation'] = np.int8(AnswerALSdata_results['SOD1 mutation'])
    y = AnswerALSdata_results['est_progress_speed']
    model = sm.OLS(y, X).fit()
    print(model.summary())

    plt.figure()
    sns.lmplot(data=AnswerALSdata_results, x='Age At Symptom Onset', y='est_progress_speed', hue='SOD1 mutation', col='Sex', ci=None)
    plt.savefig("SOD1_speed_regression.svg", dpi=300)
    plt.show()

    print('--- SOD1 regression ---')
    print(X)
    predicted_values = model.predict(X)
    residuals = AnswerALSdata_results['est_progress_speed'] - predicted_values
    plt.figure(figsize=(5, 5))
    plt.scatter(predicted_values, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values')
    plt.savefig("SDO1_speed_residuals.svg", dpi=300)
    plt.show()

    # Q-Q plot
    plt.figure(figsize=(5, 5))
    probplot(residuals, plot=plt)
    plt.title('Q-Q Plot of Residuals')
    plt.savefig("SOD1_speed_residualsQQ.svg", dpi=300)
    plt.show()

    coefficients = model.params
    conf_int = model.conf_int(alpha=0.05)
    # Plotting
    plt.rcParams['figure.subplot.bottom'] = 0.40
    plt.rcParams['figure.subplot.left'] = 0.18
    plt.figure(figsize=(4.8, 7))
    plt.errorbar(coefficients.index, coefficients.values, yerr=[coefficients.values - conf_int[0], conf_int[1] - coefficients.values], fmt='o')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xticks(rotation=60)
    plt.ylabel("Coefficient")
    plt.savefig("SOD1_speed_confit.svg", dpi=300)
    plt.show()
    plt.rcParams['figure.subplot.bottom'] = 0.20
    plt.rcParams['figure.subplot.left'] = 0.20
    #-------------------------------------------------------------------------------------    

    plt.figure(figsize=(5, 5))
    means = AnswerALSdata_results.groupby('C9orf72 mutation')['est_progress_speed'].mean()
    sns.swarmplot(data=AnswerALSdata_results, x='C9orf72 mutation', y='est_progress_speed')
    plt.xlabel('C9orf72 mutation')
    plt.ylabel('Estimated progression speed')
    plt.savefig('c9orf72_speed_compare.svg', dpi=300)
    plt.show()
    t_stat, p_value = ttest_ind(AnswerALSdata_results['est_progress_speed'][AnswerALSdata_results['C9orf72 mutation']==0],AnswerALSdata_results['est_progress_speed'][AnswerALSdata_results['C9orf72 mutation']==1])
    print('p_value: '+str(p_value))

    ########## mutation and cluster #############
    clusters = AnswerALSdata_results['cluster']
    # Pie chart data
    labels = ['', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
    sizes = [0, len(np.where(clusters==1)[0]), len(np.where(clusters==2)[0]), len(np.where(clusters==3)[0]), len(np.where(clusters==4)[0]), len(np.where(clusters==5)[0]), len(np.where(clusters==6)[0])]  # The percentages of each part of the pie chart
    plt.figure(figsize=(5, 5))
    # Create pie chart
    plt.pie(sizes, labels=labels, startangle=140)
    plt.axis('equal')
    plt.savefig('cluster_ratio.jpg', dpi=300)
    plt.savefig('cluster_ratio.svg')
    plt.show()

    c9orf72_carrier_clusters = AnswerALSdata_results[AnswerALSdata_results['C9orf72 mutation']==1]['cluster']
    sizes = [0, len(np.where(c9orf72_carrier_clusters==1)[0]), len(np.where(c9orf72_carrier_clusters==2)[0]), len(np.where(c9orf72_carrier_clusters==3)[0]), len(np.where(c9orf72_carrier_clusters==4)[0]), len(np.where(c9orf72_carrier_clusters==5)[0]), len(np.where(c9orf72_carrier_clusters==6)[0])]  # The percentages of each part of the pie chart  # The percentages of each part of the pie chart
    plt.figure(figsize=(5, 5))
    # Create pie chart
    plt.pie(sizes, labels=labels, startangle=140)
    plt.axis('equal')
    plt.savefig('c9orf72_cluster_ratio.jpg', dpi=300)
    plt.savefig('c9orf72_cluster_ratio.svg')
    plt.show()

    atxn2_carrier_clusters = AnswerALSdata_results[AnswerALSdata_results['ATXN2 mutation']==1]['cluster']
    sizes = [0, len(np.where(atxn2_carrier_clusters==1)[0]), len(np.where(atxn2_carrier_clusters==2)[0]), len(np.where(atxn2_carrier_clusters==3)[0]), len(np.where(atxn2_carrier_clusters==4)[0]), len(np.where(atxn2_carrier_clusters==5)[0]), len(np.where(atxn2_carrier_clusters==6)[0])]
    plt.figure(figsize=(5, 5))
    # Create pie chart
    plt.pie(sizes, labels=labels, startangle=140)
    plt.axis('equal')
    plt.savefig('ATXN2_cluster_ratio.jpg', dpi=300)
    plt.savefig('ATXN2_cluster_ratio.svg')
    plt.show()

    sod1_carrier_clusters = AnswerALSdata_results[AnswerALSdata_results['SOD1 mutation']==1]['cluster']
    sizes = [0, len(np.where(sod1_carrier_clusters==1)[0]), len(np.where(sod1_carrier_clusters==2)[0]), len(np.where(sod1_carrier_clusters==3)[0]), len(np.where(sod1_carrier_clusters==4)[0]), len(np.where(sod1_carrier_clusters==5)[0]), len(np.where(sod1_carrier_clusters==6)[0])]
    plt.figure(figsize=(5, 5))
    # Create pie chart
    plt.pie(sizes, labels=labels, startangle=140)
    plt.axis('equal')
    plt.savefig('SOD1_cluster_ratio.jpg', dpi=300)
    plt.savefig('SOD1_cluster_ratio.svg')
    plt.show()

    print('--------------------------')

    le = LabelEncoder()
    AnswerALSdata_results['C9orf72 mutation'] = le.fit_transform(AnswerALSdata_results['C9orf72 mutation'])

    ALS_omics_analysis(AnswerALSdata_results)

if __name__ == "__main__":
    main()

