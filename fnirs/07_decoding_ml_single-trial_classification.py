# =============================================================================
# fNIRS Classification
# Use standard features and simple linear models
# with nested cross-validation per subject
# combine cross_val with Grid_Search_CV
# =============================================================================

# import necessary modules
import os
import numpy as np
import pandas as pd
os.chdir('../..')
from toolbox import config_analysis, helper_ml
# =============================================================================
# Paths and Variables
# =============================================================================
data_directory = os.path.join(config_analysis.project_directory, 'sourcedata')
epoch_length = 10
include_silence = 'include_silence' #'_include_silence'
analysis_settings = 'fNIRS_decoding_epoch_length_{}_{}'.format(epoch_length, include_silence)
df_epochs = pd.read_csv(os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_preproc', analysis_settings, 'df_epochs.csv'), decimal = ',', sep = ';', header = 0)
save_directory = os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_decoding', analysis_settings)
os.makedirs("{}".format(save_directory), exist_ok=True)
###############################################################################
# Load Data
###############################################################################
#Load epochs for all subjects
df_epochs = pd.read_csv(os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_preproc', analysis_settings, 'df_epochs.csv'), decimal = ',', sep = ';', header = 0)
# =============================================================================
# define necessary variables
# =============================================================================
subj_list = sorted(df_epochs['ID'].unique())
ROI = 'full'
classifier_list = ['LDA', 'Dummy']
for feature_selection in  helper_ml.get_pipeline_settings().keys():
    print(ROI, feature_selection)
    pipeline_settings = helper_ml.get_pipeline_settings()[feature_selection]
    np.random.seed(pipeline_settings['SEED'])
    # Extract Features and create group variable for features

    if ('topographical_analysis' in feature_selection):
        for feat in [["average"], ["max"], ["peak2peak"]]:
            labels, df_features, features_dict = helper_ml.feature_extraction(df_epochs, pipeline_settings['hrf'],
                                                                              pipeline_settings['group_interval'],
                                                                              'ID', feat,
                                                                              "condition")
            if len(feat) == 6:
                feat = 'all'
            elif len(feat) == 1:
                feat = feat[0]
            for col in ['epoch', 'time']:
                if col in df_features.columns:
                    df_features = df_features.drop(col, axis=1)
            for iclassifier, classifier in enumerate(classifier_list):
                dict_gs = helper_ml.get_dict_classifier_GridSearch(pipeline_settings['SEED'], pipeline_settings['n_jobs'])[
                    classifier]
                save = "{}/{}/{}/{}/{}".format(save_directory, feature_selection,
                                               ROI + '_' + pipeline_settings['hrf'], feat, classifier)
                os.makedirs("{}".format(save), exist_ok=True)

                for isub, subj in enumerate(subj_list):
                    X = df_features.loc[df_features['ID'] == subj, df_features.columns != 'ID']
                    y = labels[X.index]
                    if len(pipeline_settings['contrast']) == 2:
                        # two class classification
                        indices_drop = y[~y.isin(pipeline_settings['contrast'])].index
                        y = y[y.isin(pipeline_settings['contrast'])]
                        X = X.drop(indices_drop, axis=0)
                        class_name = pipeline_settings['contrast'][0] + '_vs_' + pipeline_settings['contrast'][1]

                    elif len(pipeline_settings['contrast']) == 4:
                        # two class classification
                        indices_drop = y[~y.isin(pipeline_settings['contrast'])].index
                        y = y[y.isin(pipeline_settings['contrast'])]
                        X = X.drop(indices_drop, axis=0)
                        class_name = pipeline_settings['contrast'][0] + '_vs_' + pipeline_settings['contrast'][1] + \
                                     pipeline_settings['contrast'][2] + '_vs_' + pipeline_settings['contrast'][3]

                    if not os.path.exists(os.path.join(save, 'patterns_without_SFS_{}_{}.mat'.format(subj,class_name))):
                        helper_ml.single_trial_decoding_linear_models_get_patterns(subj, X, y, classifier, pipeline_settings, dict_gs, save)
                    else:
                        print('Already processed subject: %s, %s, %s, %s with %s' % (
                            subj, ROI, classifier, feat, class_name))


                    if not os.path.exists(os.path.join(save, 'scores_without_SFS_{}_{}.mat'.format(subj,class_name))):
                        print('processing subject: %s, %s, %s, %s with %s' % (
                            subj, ROI, feature_selection, classifier, class_name))
                        count_subj = helper_ml.single_trial_decoding_linear_models(subj, X, y,
                                                                                   classifier, pipeline_settings,
                                                                                   dict_gs, save)
                        count_subj.to_csv(os.path.join(save, 'count_subjs.csv'),
                                          header=(not os.path.exists(os.path.join(save, 'count_subjs.csv'))),
                                          index=False,
                                          decimal=',', sep=';', mode='a')
                    else:
                        print('Already processed subject: %s, %s, %s, %s with %s' % (
                            subj, ROI, classifier, feat, class_name))

    else:
        labels, df_features, _ = helper_ml.feature_extraction(df_epochs, pipeline_settings['hrf'], pipeline_settings['group_interval'], 'ID', pipeline_settings['fe'],
                                                                      "condition")
        feat = 'all'

        for col in ['epoch', 'time']:
            if col in df_features.columns:
                df_features = df_features.drop(col, axis=1)
        for iclassifier, classifier in enumerate(classifier_list):
            dict_gs = helper_ml.get_dict_classifier_GridSearch(pipeline_settings['SEED'], pipeline_settings['n_jobs'])[classifier]
            save = "{}/{}/{}/{}/{}".format(save_directory, feature_selection, ROI + '_' + pipeline_settings['hrf'], feat, classifier)
            os.makedirs("{}".format(save), exist_ok=True)
            for isub, subj in enumerate(subj_list):
                print(subj)
                X = df_features.loc[df_features['ID'] == subj,df_features.columns != 'ID']
                y = labels[X.index]
                if len(pipeline_settings['contrast']) == 2:
                    # two class classification
                    indices_drop = y[~y.isin(pipeline_settings['contrast'])].index
                    y = y[y.isin(pipeline_settings['contrast'])]
                    X = X.drop(indices_drop, axis=0)
                    class_name = pipeline_settings['contrast'][0] + '_vs_' + pipeline_settings['contrast'][1]

                elif len(pipeline_settings['contrast']) == 4:
                    # two class classification
                    indices_drop = y[~y.isin(pipeline_settings['contrast'])].index
                    y = y[y.isin(pipeline_settings['contrast'])]
                    X = X.drop(indices_drop, axis=0)
                    class_name = pipeline_settings['contrast'][0] + '_vs_' + pipeline_settings['contrast'][1] + pipeline_settings['contrast'][2] + '_vs_' + pipeline_settings['contrast'][3]
                if not os.path.exists(os.path.join(save, 'scores_without_SFS_{}_{}.mat'.format(subj, class_name))):
                    print('processing subject: %s, %s, %s, %s with %s' % (subj, ROI, feature_selection, classifier, class_name))
                    count_subj = helper_ml.single_trial_decoding_linear_models(subj, X, y, classifier, pipeline_settings, dict_gs, save)
                    count_subj.to_csv(os.path.join(save, 'count_subjs.csv'),
                                      header=(not os.path.exists(os.path.join(save, 'count_subjs.csv'))), index=False,
                                      decimal=',', sep=';', mode='a')
                else:
                    print('Already processed subject: %s, %s, %s with %s' % (subj, ROI, classifier, class_name))
