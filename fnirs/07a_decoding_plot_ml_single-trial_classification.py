# =============================================================================
# fNIRS WORKLOAD Classification - Plot Results
# Use standard features and simple linear models
# with nested cross-validation per subject
# combine cross_val with Grid_Search_CV
# =============================================================================

# import necessary modules
import os
import numpy as np
import pandas as pd
import mne
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
os.chdir('..')
from toolbox import helper_ml, helper_plot, config_analysis

np.random.seed(42)

# =============================================================================
# Directories
# =============================================================================
###############################################################################
# Load Data
###############################################################################
fig_format = ['svg', 'png']
plot_coefficients = True
plot_patterns = False
classifier_of_interest = 'LDA'
contrast = 'HighNeg_vs_LowNeg_vs_HighPos_vs_LowPos'
epoch_length = 10
include_silence = 'include_silence' #'_include_silence'
analysis_settings = 'fNIRS_decoding_epoch_length_{}_{}'.format(epoch_length, include_silence)
data_path = os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_decoding', analysis_settings)
save_path = os.path.join(data_path, 'plots')
os.makedirs(save_path, exist_ok=True)

exemplary_raw_haemo = mne.io.read_raw_fif(os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_preproc',analysis_settings, "exemplary_raw.fif")).load_data()
exemplary_raw_haemo.info['bads'] = []
mne.datasets.fetch_fsaverage()
colormaps = {'hbr': matplotlib.cm.RdBu_r, 'subj_hbr' : matplotlib.cm.PiYG, 'hbo': matplotlib.cm.RdBu_r, 'subj_hbo' : matplotlib.cm.PiYG_r, 'standard_error': matplotlib.cm.Reds}
subj_list_plotting = ['weighted_average'] #['standard_error', 7, 16, 'weighted_average', 'average']
features_of_interest = ['max LDA']
df_empirical_chance_level = pd.DataFrame()
for analysis in helper_ml.get_pipeline_settings().keys():
    if 'topographical' not in analysis:
        continue
        score_list = [('scores_with_SFS', '.mat', '_SFS')]

    else:
        score_list = [('scores_without_SFS', '.mat', '')]

    for specification in os.listdir(os.path.join(data_path, analysis)):
        save = os.path.join(save_path, analysis)
        os.makedirs(save, exist_ok=True)
        chroma = specification[-3:]
        if chroma == 'hbo':
            color_per_subj = '#eee8aa'
            average_color = '#a7a277'
            list_color_classifier = ['#eee8aa', '#fcc9b5', '#f194b8']
            list_color_classifier_averaged = [ '#a7a277', '#b08d7f','#a96881']
            lims_coefficients = (-0.5, 0, 0.5)
            lims_patterns  = (-0.5, 0, 0.5)
        elif chroma == 'hbr':
            color_per_subj = '#00cee2'
            average_color = '#00b4c5'
            list_color_classifier = ['#00cee2', '#0079f2', '#00df92']
            list_color_classifier_averaged = [ '#00b4c5', '#0073e6', '#00bf7d']
            lims_coefficients = (-0.5, 0, 0.5)
            lims_patterns = (-0.5, 0, 0.5)
        df_full = pd.DataFrame()
        list_features = [feature for feature in os.listdir(os.path.join(data_path, analysis, specification)) if '.' not in feature]
        for feature in list_features:
            print(feature)
            # =============================================================================
            # Plot per Subject
            # =============================================================================
            if os.path.exists(os.path.join(save_path, analysis,  feature, 'df_boot.csv')):
                df_boot = pd.read_csv(os.path.join(save_path, analysis, feature, 'df_boot.csv'),
                                      header=0,
                                      decimal=',', sep=';')
            else:
                df_boot = helper_ml.create_df_boot_scores(data_path, save_path, analysis, specification,
                                                          feature,
                                                          score_list, contrast)
            # rename subj to start at 1
            dict_rename_subj = dict(zip(list(df_boot['subj'].unique()), list(
                ['sub-0' + str(subj_num) if len(str(subj_num)) == 1 else 'sub-' + str(subj_num) for subj_num in
                 range(1, len(df_boot['subj'].unique()) + 1)])))
            df_boot['subj'] = df_boot['subj'].replace(dict_rename_subj)
            # add average
            df_average = df_boot.copy()
            df_average['subj'] = 'average'
            df_boot = pd.concat((df_boot, df_average))
            df_boot = df_boot.reset_index(drop=True)
            # create df_boot in right bootstrapping format with classifiers as columns
            df_boot = df_boot.pivot_table(values=['train', 'test'], index=['subj', 'fold'],
                                          columns="classifier")
            df_boot.columns = ['_'.join((col[1], col[0])) for col in df_boot.columns]
            df_boot = df_boot.reset_index()

            # take only test set
            df_plt = df_boot.loc[:, [col for col in df_boot.columns if 'test' in col] + ['subj', 'fold']]
            df_plt.columns = [name[: - len('_test')] if '_test' in name else name for name in df_plt.columns]
            df_plt['feature'] = feature
            df_full = pd.concat((df_full, df_plt))
        #include feature column in column row - from long to wide
        df_wide = pd.pivot(df_full, index=['subj', 'fold'], columns='feature').reset_index(drop = False)
        df_wide.columns = [tup[1] + ' ' + tup[0] if len(tup[1]) > 1 else tup[0] for tup in df_wide.columns]
        #reshape to have all variables in one column - from wide to long

        df_long = df_wide.melt(id_vars=['subj', 'fold'])
        df_long = df_long.loc[~df_long['variable'].isin([dummy for dummy in list(df_long['variable'].unique()) if 'Dummy' in dummy])]
        # get overall empirical chance level
        empirical_chance_level = helper_plot.bootstrapping(
            df_full.loc[df_full['subj'] != 'average', 'Dummy'].values,
            sample_size=None, numb_iterations=5000, alpha=0.95,
            plot_hist=False, as_dict=True, func='mean')
        empirical_chromopore = pd.DataFrame.from_dict(empirical_chance_level, orient= 'index').T
        empirical_chromopore['Chromophore'] = chroma
        df_empirical_chance_level = pd.concat((df_empirical_chance_level, empirical_chromopore))
        print('EMPIRICAL CHANCE LEVEL', analysis, empirical_chance_level)
        list_color = list_color_classifier * len(os.listdir(os.path.join(data_path, analysis, specification)))

        ### FOR ALL THREE CLASSIFIERS
        #color_lines_legend.append(Line2D([0], [0], color='salmon', lw=3))
        #        lst_groups.append('Dummy - Chance Level')
        n_classifiers = len([col for col in df_full.columns if col not in ['Dummy', 'subj', 'fold', 'feature']])
        x_labels = list(np.repeat(list_features,n_classifiers))
        x_labels = [''] * len(x_labels)
        if n_classifiers > 1:
            x_labels[1::n_classifiers] = list_features
            legend_lines = []
            for icl in range(0, n_classifiers):
                legend_lines.append(Line2D([0], [0], color=list_color[icl], lw=3))
        else:
            x_labels = list_features
            legend_lines = [Line2D([0], [0], color=list_color[0], lw=3)]

        fig, df_boo = helper_plot.plot_single_trial_bootstrapped_boxplots(
            sorted(df_long['variable'].unique()), df_long, 'variable', None,
            ['value'],
            boot='mean', boot_size=5000,
            title='Four Class LDA-Decoding: Negative - Positive Emotion High - Low Load (' + chroma.replace(
                '_', ' ').upper() + ')', lst_color=list_color, ncolor_group=1,
            fwr_correction=True,
            x_labels=x_labels, chance_level=0.25,
            legend=False,
            n_col=1,
            figsize=(13, 7),
            empirical_chance_level=empirical_chance_level, fs = 22, axtitle = False)

        df_boo.to_csv(os.path.join(save, 'df_boot_LDA_averaged_class_per_feature.csv'), header=True,
                      index=True,
                      decimal=',', sep=';')
        for end in fig_format:
            fig.savefig("{}/df_boot_LDA_averaged_class_per_feature.{}".format(save, end))
        plt.clf()
        ### FOR LDA CLASSIFIERS without ALL PER SUBJECT
        df_classifier_of_interest = df_wide.loc[:, [col for col in df_wide.columns if (classifier_of_interest in col) or (col in ['subj', 'fold'])]]
        df_classifier_of_interest = df_classifier_of_interest.melt(id_vars=['subj', 'fold'])
        list_color_subj = list_color_classifier * (len(df_classifier_of_interest['subj'].unique()) - 1) + list_color_classifier_averaged
        x_labels = list(np.repeat(list(df_classifier_of_interest['subj'].unique()),len(list_features)))
        x_labels = [''] * len(x_labels)

        x_labels[1::len(list_features)] = list(df_classifier_of_interest['subj'].unique())[1:] + list(df_classifier_of_interest['subj'].unique())[:1]
        legend_lines = []
        for icl in range(0, len(list_features)):
            legend_lines.append(Line2D([0], [0], color=list_color[icl], lw=3))
        df_classifier_of_interest['subj'] = [isub + '_' + row.split(' ')[0] for isub, row in zip(df_classifier_of_interest['subj'], df_classifier_of_interest['variable'])]

        fig, df_boo = helper_plot.plot_single_trial_bootstrapped_boxplots(
            sorted(df_classifier_of_interest['subj'].unique())[len(list_features):] + ['average_average', 'average_max', 'average_peak2peak'], df_classifier_of_interest, 'subj', 'fold',
            ['value'],
            boot='mean', boot_size=5000,
            title='', lst_color=list_color_subj, ncolor_group=1,
            fwr_correction=True,
            x_labels=x_labels, chance_level=0.25,
            legend= (legend_lines, sorted(list_features)),
            n_col=1,
            figsize=(20, 6),
            empirical_chance_level=empirical_chance_level, fs = 16, axtitle = 'Four Class LDA-Decoding: Negative - Positive Emotion High - Low Load (' + chroma.replace(
                '_', ' ').upper() + ')')
        df_boo.to_csv(os.path.join(save, classifier_of_interest + '_boot_per_subj_per_feat.csv'), header=True,
                      index=True, decimal=',', sep=';')
        for end in fig_format:
            fig.savefig("{}/{}_boot_per_subj_per_feat.{}".format(save, classifier_of_interest, end))
        plt.clf()

        # uncomment for single plots per feature
        for feature in [feat for feat in sorted(df_long['variable'].unique()) if classifier_of_interest in feat]:
            if plot_coefficients:
                # =============================================================================
                # Plot Coefficients averaged over Participants
                # =============================================================================
                if os.path.exists(os.path.join(save_path, analysis, feature.split(' ')[0], 'df_coef.csv')):
                    df_coef = pd.read_csv(os.path.join(save_path, analysis, feature.split(' ')[0], 'df_coef.csv'),
                                          header=0,
                                          decimal=',', sep=';')
                else:
                    df_coef = helper_ml.create_df_coefficients(data_path, save_path, analysis, specification, feature.split(' ')[0],
                                                               contrast)

                helper_plot.plot_weights_3d_brain(df_coef, 'coef', chroma,  subj_list_plotting,
                                                  colormaps, lims_coefficients, exemplary_raw_haemo, feature.upper().replace(' ', '_'), save, True, False)
            if plot_patterns:
                # =============================================================================
                # Plot Patterns averaged over Participants
                # =============================================================================
                if os.path.exists(os.path.join(save_path, analysis, feature.split(' ')[0], 'df_patterns.csv')):
                    df_patterns = pd.read_csv(os.path.join(save_path, analysis, feature.split(' ')[0], 'df_patterns.csv'),
                                              header=0,
                                              decimal=',', sep=';')
                else:
                    df_patterns = helper_ml.create_df_patterns(data_path, save_path, analysis, specification, feature.split(' ')[0], contrast)

                helper_plot.plot_weights_3d_brain(df_patterns, 'patterns', chroma, subj_list_plotting,
                                                  colormaps, lims_patterns, exemplary_raw_haemo, 'Patterns_' + feature.upper().replace(' ', '_'), save, False, False)
df_empirical_chance_level.to_csv(os.path.join(save_path, 'df_empirical_chance_level.csv'), header = True, index = False, decimal=',', sep = ';')
