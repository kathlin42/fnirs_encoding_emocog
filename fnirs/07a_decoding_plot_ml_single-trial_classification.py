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
import mne_nirs
mne.viz.set_3d_backend("pyvista")  # pyvistaqt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
os.chdir('..')
from toolbox import helper_ml, helper_plot, config_analysis

np.random.seed(42)

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'~qgis directory\apps\Qt5\plugins'
os.environ['PATH'] += r';~qgis directory\apps\qgis\bin;~qgis directory\apps\Qt5\bin'

# =============================================================================
# Directories
# =============================================================================
###############################################################################
# Load Data
###############################################################################
fig_format = ['svg', 'png']
plot_coefficients = True
plot_patterns = True
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

for analysis in helper_ml.get_pipeline_settings().keys():
    if 'topographical' in analysis:
        score_list = [('scores_without_SFS', '.mat', '')]

    else:
        score_list = [('scores_with_SFS', '.mat', '_SFS')]
    for specification in os.listdir(os.path.join(data_path, analysis)):
        save = os.path.join(save_path, analysis, specification)
        os.makedirs(save, exist_ok=True)
        chroma = specification[-3:]
        if chroma == 'hbo':
            colormap = matplotlib.cm.bwr
            color_per_subj = '#f194b8'
            average_color = '#a96881'
            list_color_classifier = ['#f194b8', '#fcc9b5', '#eee8aa']
            list_color_classifier_averaged = ['#a96881', '#b08d7f', '#a7a277']
            lims_coefficients = (-0.5, 0, 0.5)
            lims_patterns  = (-0.2, 0, 0.2)
        elif chroma == 'hbr':
            colormap = matplotlib.cm.PiYG_r
            color_per_subj = '#00df92'
            average_color = '#00bf7d'
            list_color_classifier = ['#00df92', '#00cee2', '#0079f2']
            list_color_classifier_averaged = ['#00bf7d', '#00b4c5', '#0073e6']
            lims_coefficients = (-0.2, 0, 0.2)
            lims_patterns = (-0.2, 0, 0.2)
        df_full = pd.DataFrame()
        list_features = [feature for feature in os.listdir(os.path.join(data_path, analysis, specification)) if '.' not in feature]
        for feature in list_features:
            print(feature)
            # =============================================================================
            # Plot per Subject
            # =============================================================================
            if os.path.exists(os.path.join(save_path, analysis, specification, feature, 'df_boot.csv')):
                df_boot = pd.read_csv(os.path.join(save_path, analysis, specification, feature, 'df_boot.csv'),
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
        df_long = df_long.loc[~df_long['variable'].isin([LR for LR in list(df_long['variable'].unique()) if 'LR' in LR])]
        # get overall empirical chance level
        empirical_chance_level = helper_plot.bootstrapping(
            df_full.loc[df_full['subj'] != 'average', 'Dummy'].values,
            sample_size=None, numb_iterations=5000, alpha=0.95,
            plot_hist=False, as_dict=True, func='mean')
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
            df_feat = df_long.loc[df_long['variable'] == feature]
            df_feat = df_feat.rename(columns= {'value': feature.split(' ')[0]})
            if plot_coefficients:
                # =============================================================================
                # Plot Coefficients averaged over Participants
                # =============================================================================
                if os.path.exists(os.path.join(save_path, analysis, specification, feature.split(' ')[0], 'df_coef.csv')):
                    df_coef = pd.read_csv(os.path.join(save_path, analysis, specification, feature.split(' ')[0], 'df_coef.csv'),
                                          header=0,
                                          decimal=',', sep=';')
                else:
                    df_coef = helper_ml.create_df_coefficients(data_path, save_path, analysis, specification, feature.split(' ')[0],
                                                               contrast)
                for classifier in df_coef['classifier'].unique():
                    for con in df_coef['con'].unique():
                        df_coef_classifier = df_coef.loc[(df_coef['classifier'] == classifier) & (df_coef['con'] == con), df_coef.columns != 'classifier']
                        df_coef_classifier['Source'] = [int(ss.split('_')[0]) for ss in
                                                        [s.split('S')[1] for s in df_coef_classifier.features]]
                        df_coef_classifier['Detector'] = [int(dd.split(' ')[0]) for dd in
                                                          [d.split('D')[1] for d in df_coef_classifier.features]]

                        con_summary = pd.DataFrame(
                            {'ID': df_coef_classifier.subj.values,
                             'Contrast': [con] * len(df_coef_classifier.subj.values),
                             'ch_name': [ch.rsplit(chroma, 1)[0] + chroma for ch in df_coef_classifier.features],
                             'Chroma': [chroma] * len(df_coef_classifier.subj.values),
                             'coef': df_coef_classifier.coef.values})
                        con_model = smf.mixedlm("coef ~ -1 + ch_name:Chroma", con_summary, groups=con_summary["ID"]).fit(
                            method='nm')
                        df_con_model = mne_nirs.statistics.statsmodels_to_results(con_model)
                        for subj in df_coef['subj'].unique().tolist() + ['average']:
                            if subj != 'average':
                                coefficients = df_coef_classifier.loc[(df_coef_classifier['subj'] == subj)].groupby(['features']).mean(numeric_only = True).reset_index()
                            else:
                                coefficients = df_coef_classifier.groupby(['features']).mean(numeric_only = True).reset_index()
                            assert len(coefficients) == len(df_con_model['Coef.'].values)
                            df_con_model['Coef.'] = coefficients['coef'].values
                            df_con_model['Source'] = coefficients['Source'].values
                            df_con_model['Detector'] = coefficients['Detector'].values
                            df_con_model = df_con_model.sort_values(by=['Source', 'Detector'], ascending=[True, True])

                            mne.datasets.fetch_fsaverage()
                            # Cortical Surface Projections for Contrasts
                            for view in ['rostral', 'lateral']:
                                # view = 'lateral'
                                if view == 'lateral':
                                    hemis = ['lh', 'rh']
                                    colorbar = False
                                else:
                                    hemis = ['both']
                                    colorbar = True
                                for hemi in hemis:
                                    # hemi = 'both'
                                    save_name = specification.upper() + '_' + feature.upper().replace(' ', '_')
                                    brain = mne_nirs.visualisation.plot_glm_surface_projection(
                                        exemplary_raw_haemo.copy().pick(picks=chroma),
                                        statsmodel_df=df_con_model, picks=chroma,
                                        view=view, hemi=hemi, clim={'kind': 'value', 'lims': lims_coefficients},
                                        colormap=colormap, colorbar=colorbar, size=(800, 700))
                                    if subj != 'average':
                                        save_brain_plot = os.path.join(save, save_name, 'subject_level', con)
                                        os.makedirs(save_brain_plot, exist_ok=True)
                                        brain.save_image(
                                            "{}/sub-{}_{}_{}.png".format(save_brain_plot, str(subj), view, hemi))
                                        brain.close()
                                    else:
                                        save_brain_plot = os.path.join(save, save_name, subj, con)
                                        os.makedirs(save_brain_plot, exist_ok=True)
                                        brain.save_image(
                                            "{}/average-{}_{}.png".format(save_brain_plot, view, hemi))
                                        brain.close()
            if plot_patterns:
                # =============================================================================
                # Plot Patterns averaged over Participants
                # =============================================================================
                if os.path.exists(os.path.join(save_path, analysis, specification, feature.split(' ')[0], 'df_patterns.csv')):
                    df_patterns = pd.read_csv(os.path.join(save_path, analysis, specification, feature.split(' ')[0], 'df_patterns.csv'),
                                              header=0,
                                              decimal=',', sep=';')
                else:
                    df_patterns = helper_ml.create_df_patterns(data_path, save_path, analysis, specification, feature.split(' ')[0],
                                                               contrast)
                for classifier in df_patterns['classifier'].unique():
                    for con in df_patterns['con'].unique():
                        print(classifier)
                        df_patterns_classifier = df_patterns_classifier.loc[(df_patterns_classifier['classifier'] == classifier) & (df_patterns_classifier['con'] == con), df_patterns_classifier.columns != 'classifier']

                        df_patterns_classifier['Source'] = [int(ss.split('_')[0]) for ss in
                                                            [s.split('S')[1] for s in df_patterns_classifier.features]]
                        df_patterns_classifier['Detector'] = [int(dd.split(' ')[0]) for dd in
                                                              [d.split('D')[1] for d in df_patterns_classifier.features]]

                        con_summary = pd.DataFrame({'ID': df_patterns_classifier.subj.values,
                                                    'Contrast': [con] * len(df_patterns_classifier.subj.values),
                                                    'ch_name': [ch.rsplit(chroma, 1)[0] + chroma for ch in
                                                                df_patterns_classifier.features],
                                                    'Chroma': [chroma] * len(df_patterns_classifier.subj.values),
                                                    'patterns': df_patterns_classifier.patterns.values})
                        con_model = smf.mixedlm("patterns ~ -1 + ch_name:Chroma", con_summary, groups=con_summary["ID"]).fit(
                            method='nm')
                        df_con_model = mne_nirs.statistics.statsmodels_to_results(con_model)
                        for subj in df_coef['subj'].unique().tolist() + ['average']:
                            if subj != 'average':
                                coefficients = df_patterns_classifier.loc[(df_patterns_classifier['subj'] == subj)].groupby(
                                    ['features']).mean(numeric_only=True).reset_index()
                            else:
                                coefficients = df_patterns_classifier.groupby(['features']).mean(
                                    numeric_only=True).reset_index()

                            assert len(coefficients) == len(df_con_model['Coef.'].values)
                            df_con_model['Coef.'] = coefficients['patterns'].values
                            df_con_model['Source'] = coefficients['Source'].values
                            df_con_model['Detector'] = coefficients['Detector'].values
                            df_con_model = df_con_model.sort_values(by=['Source', 'Detector'], ascending=[True, True])

                            mne.datasets.fetch_fsaverage()
                            # Cortical Surface Projections for Contrasts
                            for view in ['rostral', 'lateral']:
                                # view = 'lateral'
                                if view == 'lateral':
                                    hemis = ['lh', 'rh']
                                    colorbar = False
                                else:
                                    hemis = ['both']
                                    colorbar = True
                                for hemi in hemis:
                                    # hemi = 'both'
                                    save_name = 'Patterns_' + specification.upper() + '_' + feature.upper().replace(' ', '_')
                                    brain = mne_nirs.visualisation.plot_glm_surface_projection(
                                        exemplary_raw_haemo.copy().pick(picks=chroma),
                                        statsmodel_df=df_con_model, picks=chroma,
                                        view=view, hemi=hemi, clim={'kind': 'value', 'lims': lims_patterns},
                                        colormap=colormap, colorbar=colorbar, size=(800, 700))

                                    if subj != 'average':
                                        save_brain_plot = os.path.join(save, save_name, 'subject_level', con)
                                        os.makedirs(save_brain_plot, exist_ok=True)
                                        brain.save_image(
                                            "{}/sub-{}_{}_{}.png".format(save_brain_plot, str(subj), view, hemi))
                                        brain.close()
                                    else:
                                        save_brain_plot = os.path.join(save, save_name, subj, con)
                                        os.makedirs(save_brain_plot, exist_ok=True)
                                        brain.save_image(
                                            "{}/average-{}_{}.png".format(save_brain_plot, view, hemi))
                                        brain.close()
