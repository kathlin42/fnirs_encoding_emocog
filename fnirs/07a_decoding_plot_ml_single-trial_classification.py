# =============================================================================
# fNIRS WORKLOAD Classification - Plot Results
# Use standard features and simple linear models
# with nested cross-validation per subject
# combine cross_val with Grid_Search_CV
# =============================================================================

# import necessary modules
import os
import warnings
import numpy as np
import pandas as pd

from scipy.io import loadmat
import mne
import mne_nirs
mne.viz.set_3d_backend("pyvista")  # pyvistaqt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
from natsort import natsort_keygen
import statsmodels.formula.api as smf
os.chdir(os.path.join('..', os.getcwd()))
from toolbox import (helper_ml, helper_plot)

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
bids_directory = os.path.join("R:/NirAcademy_81024520/!Ergebnisse/03_NIRCADEMY_1", "04_nircademy_bids")

group_variable = 'epoch'
contrast = 'OneBack_LW_vs_ThreeBack_HW'

data_path = os.path.join(bids_directory, 'derivatives', 'fNIRS', 'single_trial_decoding', group_variable)
save_path = os.path.join(bids_directory, 'derivatives', 'fNIRS', 'single_trial_decoding', group_variable, 'plots')
if not os.path.exists(save_path):
    os.makedirs(save_path)

exemplary_raw_haemo = mne.io.read_raw_fif(os.path.join(bids_directory, 'derivatives', 'fnirs','configurations',  "exemplary_raw.fif")).load_data()
exemplary_raw_haemo.info['bads'] = []


for analysis in [folder for folder in os.listdir(data_path) if folder not in ['plots', 'linear_model','topographical_analysis_hbo+hbr', 'linear_model_k_10','linear_model_k_20']]:
    if 'topographical' in analysis:
        score_list = [('scores_without_SFS', '.mat', '')]

    else:
        score_list = [('scores_without_SFS', '.mat', ''), ('scores_with_SFS', '.mat', '_SFS')]

    for specification in os.listdir(os.path.join(data_path, analysis)):
        save = os.path.join(save_path, analysis, specification)
        if not os.path.exists(save):
            os.makedirs(save)
        chroma = specification[-3:]
        if chroma == 'hbo':
            colormap = matplotlib.cm.bwr
            color_per_subj = 'peachpuff'
            average_color = 'sandybrown'
            list_color_classifier = ['peachpuff', 'cornflowerblue', 'mediumaquamarine']
            list_color_classifier_averaged = ['sandybrown', 'royalblue', 'darkcyan']
            lims_coefficients = (-0.5, 0, 0.5)
            lims_patterns  = (-0.2, 0, 0.2)
        elif chroma == 'hbr':
            colormap = matplotlib.cm.PiYG_r
            color_per_subj = 'palegreen'
            average_color = 'mediumseagreen'
            list_color_classifier = ['palegreen', 'mediumturquoise', 'plum']
            list_color_classifier_averaged = ['seagreen', 'darkslategrey', 'mediumorchid']
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
                df_boot = helper_ml.create_df_boot_SFS_scores(data_path, save_path, analysis, specification,
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
            df_boot = df_boot.append(df_average)
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
            df_full = df_full.append(df_plt)
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
        x_labels = list(np.repeat(list_features, 3))
        x_labels = [''] * len(x_labels)
        x_labels[1::3] = list_features

        fig, df_boo = helper_plot.plot_single_trial_bootstrapped_boxplots(
            sorted(df_long['variable'].unique()), df_long, 'variable', None,
            ['value'],
            boot='mean', boot_size=5000,
            title='Performance per Classifier and Feature - ' + chroma.replace(
                '_', ' ').upper(), lst_color=list_color, ncolor_group=1,
            fwr_correction=True,
            x_labels=x_labels, chance_level=True,
            legend=([Line2D([0], [0], color=list_color[0], lw=3),
                     Line2D([0], [0], color=list_color[1], lw=3),
                     Line2D([0], [0], color=list_color[2], lw=3)],
                    ['LDA', 'SVM', 'xgBoost']),
            n_col=1,
            figsize=(13, 7),
            empirical_chance_level=empirical_chance_level, fs = 22, axtitle = False)

        df_boo.to_csv(os.path.join(save, 'df_boot_over_classifier_and_feature.csv'), header=True,
                      index=True,
                      decimal=',', sep=';')
        for end in fig_format:
            fig.savefig("{}/df_boot_over_classifier_and_feature.{}".format(save, end))
        plt.clf()

        ### FOR LDA CLASSIFIERS without ALL PER SUBJECT
        df_classifier_of_interest = df_wide.loc[:, [col for col in df_wide.columns if (classifier_of_interest in col) or (col in ['subj', 'fold'])]]
        list_color_subj = [color_per_subj] * (len(df_classifier_of_interest['subj'].unique()) - 1) + [average_color]
        fig, df_boo = helper_plot.plot_single_trial_bootstrapped_boxplots(
            sorted(df_classifier_of_interest['subj'].unique())[1:] + ['average'], df_classifier_of_interest, 'subj', 'fold',
            [col for col in df_classifier_of_interest.columns if (classifier_of_interest in col) and ('all' not in col)],
            boot='mean', boot_size=5000,
            title='Classifier Performance per Participant - ' + chroma.replace(
                '_', ' ').upper(), lst_color=list_color_subj, ncolor_group=1,
            fwr_correction=True,
            x_labels=sorted(df_classifier_of_interest['subj'].unique())[1:] + ['average'], chance_level=True,
            legend= False,
            n_col=1,
            figsize=(10, 20),
            empirical_chance_level=empirical_chance_level, fs = 18, axtitle = True)
        df_boo.to_csv(os.path.join(save, classifier_of_interest + '_boot_per_subj_auc_all_single_feats.csv'), header=True,
                      index=True, decimal=',', sep=';')
        for end in fig_format:
            fig.savefig("{}/{}_boot_per_subj_auc_all_single_feats.{}".format(save, classifier_of_interest, end))
        plt.clf()

        ### FOR ALL CLASSIFIERS ONLY ALL PER SUBJECT
        df_feat_all = df_wide.loc[:, [col for col in df_wide.columns if ('Dummy' not in col) and ('LR' not in col) and
                                      (('all' in col) or (col in ['subj', 'fold']))]]
        df_feat_all = df_feat_all.melt(id_vars=['subj', 'fold'])
        df_feat_all['unique'] = [subj + ' ' + classifier[4:] for classifier, subj in zip(df_feat_all.variable, df_feat_all.subj)]
        list_color_feat_all = list_color_classifier * int((len([sub for sub in sorted(df_feat_all['unique'].unique()) if 'average' not in sub])/ len(list_color_classifier)))\
                              + list_color_classifier_averaged
        x_labels = [sub[:7] for sub in sorted(df_feat_all['unique'].unique()) if 'average' not in sub] + [sub[:8] for sub in sorted(df_feat_all['unique'].unique()) if 'average' in sub]
        x_labels = [''] * len(x_labels)
        x_labels[1::3] = [sub[:7] for sub in sorted(df_feat_all['unique'].unique()) if 'average' not in sub][1::3] + [sub[:8] for sub in sorted(df_feat_all['unique'].unique()) if 'average' in sub][1::3]
        fig, df_boo = helper_plot.plot_single_trial_bootstrapped_boxplots(
            [sub for sub in sorted(df_feat_all['unique'].unique()) if 'average' not in sub] +
            [sub for sub in sorted(df_feat_all['unique'].unique()) if 'average' in sub],
            df_feat_all, 'unique', None,
            ['value'],
            boot='mean', boot_size=5000,
            title='Classifier Performance per Participant Using All Features - ' + chroma.replace(
                '_', ' ').upper(), lst_color=list_color_feat_all, ncolor_group=1,
            fwr_correction=True,
            x_labels=x_labels,
            chance_level=True,
            legend=([Line2D([0], [0], color=list_color[0], lw=3),
                     Line2D([0], [0], color=list_color[1], lw=3),
                     Line2D([0], [0], color=list_color[2], lw=3)],
                    ['LDA', 'SVM', 'xgBoost']),
            n_col=1,
            figsize=(13, 7),
            empirical_chance_level=empirical_chance_level, fs = 22, axtitle = False)
        df_boo.to_csv(os.path.join(save, classifier_of_interest + '_boot_per_subj_auc_all_feat.csv'), header=True,
                      index=True, decimal=',', sep=';')
        for end in fig_format:
            fig.savefig("{}/{}_boot_per_subj_auc_all_feat.{}".format(save, classifier_of_interest, end))
        plt.clf()


        # uncomment for single plots per feature
        for feature in [feat for feat in sorted(df_long['variable'].unique()) if classifier_of_interest in feat]:
            df_feat = df_long.loc[df_long['variable'] == feature]
            df_feat = df_feat.rename(columns= {'value': feature[:-4]})
            if False:
                list_color_subj = ['peachpuff'] * (len(df_feat['subj'].unique())-1) + ['salmon']
                fig, df_boo = helper_plot.plot_single_trial_bootstrapped_boxplots(
                    sorted(df_plt['subj'].unique())[1:] + ['average'], df_feat, 'subj', 'fold',
                    [feature[:-4]],
                    boot='mean', boot_size=5000,
                    title='Classifier Performance - ' + chroma.replace(
                        '_', ' ').upper(), lst_color=list_color_subj, ncolor_group=1,
                    fwr_correction=True,
                    x_labels=sorted(df_plt['subj'].unique())[1:] + ['average'], chance_level=True, legend=False,
                    n_col=1,
                    figsize=(10, 8),
                    empirical_chance_level=empirical_chance_level)
                df_boo.to_csv(os.path.join(save, classifier_of_interest + '_boot_per_subj_auc_feat_' + feature[:-4] + '.csv'), header=True,
                              index=True, decimal=',', sep=';')
                for end in fig_format:
                    fig.savefig("{}/" + classifier_of_interest + "_boot_per_subj_auc_feat_{}.{}".format(save, feature[:-4], end))
                plt.clf()

            if feature[:-4] == 'all':
                continue
            if plot_coefficients:
                # =============================================================================
                # Plot Coefficients averaged over Participants
                # =============================================================================
                if os.path.exists(os.path.join(save_path, analysis, specification, feature[:-4], 'df_coef.csv')):
                    df_coef = pd.read_csv(os.path.join(save_path, analysis, specification, feature[:-4], 'df_coef.csv'),
                                          header=0,
                                          decimal=',', sep=';')
                else:
                    df_coef = helper_ml.create_df_coefficients(data_path, save_path, analysis, specification, feature[:-4],
                                                               contrast)
                for classifier in df_coef['classifier'].unique():
                    df_coef_classifier = df_coef.loc[df_coef['classifier'] == classifier, df_coef.columns != 'classifier']
                    df_coef_classifier['Source'] = [int(ss.split('_')[0]) for ss in
                                                    [s.split('S')[1] for s in df_coef_classifier.features]]
                    df_coef_classifier['Detector'] = [int(dd.split(' ')[0]) for dd in
                                                      [d.split('D')[1] for d in df_coef_classifier.features]]

                    con_summary = pd.DataFrame(
                        {'ID': df_coef_classifier.subj.values, 'Contrast': [contrast] * len(df_coef_classifier.subj.values),
                         'ch_name': [ch.rsplit(chroma, 1)[0] + chroma for ch in df_coef_classifier.features],
                         'Chroma': [chroma] * len(df_coef_classifier.subj.values),
                         'coef': df_coef_classifier.coef.values})
                    con_model = smf.mixedlm("coef ~ -1 + ch_name:Chroma", con_summary, groups=con_summary["ID"]).fit(
                        method='nm')
                    df_con_model = mne_nirs.statistics.statsmodels_to_results(con_model)
                    coefficients = df_coef_classifier.groupby(['features']).mean().reset_index()
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
                            save_name = specification.upper() + '_' + feature.upper()
                            brain = mne_nirs.visualisation.plot_glm_surface_projection(
                                exemplary_raw_haemo.copy().pick(picks=chroma),
                                statsmodel_df=df_con_model, picks=chroma,
                                view=view, hemi=hemi, clim={'kind': 'value', 'lims': lims_coefficients},
                                colormap=colormap, colorbar=colorbar, size=(800, 700))

                            brain.save_image(
                                "{}/{}_{}_{}_{}_{}.png".format(save, classifier, save_name, contrast, view, hemi))
                            brain.close()
            if plot_patterns:
                # =============================================================================
                # Plot Patterns averaged over Participants
                # =============================================================================
                if os.path.exists(os.path.join(save_path, analysis, specification, feature[:-4], 'df_patterns.csv')):
                    df_patterns = pd.read_csv(os.path.join(save_path, analysis, specification, feature[:-4], 'df_patterns.csv'),
                                              header=0,
                                              decimal=',', sep=';')
                else:
                    df_patterns = helper_ml.create_df_patterns(data_path, save_path, analysis, specification, feature[:-4],
                                                               contrast)
                for classifier in df_patterns['classifier'].unique():
                    print(classifier)
                    df_patterns_classifier = df_patterns.loc[
                        df_patterns['classifier'] == classifier, df_patterns.columns != 'classifier']
                    df_patterns_classifier['Source'] = [int(ss.split('_')[0]) for ss in
                                                        [s.split('S')[1] for s in df_patterns_classifier.features]]
                    df_patterns_classifier['Detector'] = [int(dd.split(' ')[0]) for dd in
                                                          [d.split('D')[1] for d in df_patterns_classifier.features]]

                    con_summary = pd.DataFrame({'ID': df_patterns_classifier.subj.values,
                                                'Contrast': [contrast] * len(df_patterns_classifier.subj.values),
                                                'ch_name': [ch.rsplit(chroma, 1)[0] + chroma for ch in
                                                            df_patterns_classifier.features],
                                                'Chroma': [chroma] * len(df_patterns_classifier.subj.values),
                                                'patterns': df_patterns_classifier.patterns.values})
                    con_model = smf.mixedlm("patterns ~ -1 + ch_name:Chroma", con_summary, groups=con_summary["ID"]).fit(
                        method='nm')
                    df_con_model = mne_nirs.statistics.statsmodels_to_results(con_model)
                    coefficients = df_patterns_classifier.groupby(['features']).mean().reset_index()
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
                            save_name = 'patterns_' + specification.upper() + '_' + feature.upper()
                            brain = mne_nirs.visualisation.plot_glm_surface_projection(
                                exemplary_raw_haemo.copy().pick(picks=chroma),
                                statsmodel_df=df_con_model, picks=chroma,
                                view=view, hemi=hemi, clim={'kind': 'value', 'lims': lims_patterns},
                                colormap=colormap, colorbar=colorbar, size=(800, 700))

                            brain.save_image(
                                "{}/{}_{}_{}_{}_{}.png".format(save, classifier, save_name, contrast, view, hemi))
                            brain.close()
