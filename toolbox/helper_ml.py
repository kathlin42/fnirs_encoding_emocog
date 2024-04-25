import os
import pandas as pd
import numpy as np
import re
import mne
from mne.decoding import (LinearModel, get_coef)
from typing import List

from scipy.io import loadmat

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (StratifiedKFold, GridSearchCV, RandomizedSearchCV,
                                     RepeatedStratifiedKFold, cross_validate,
                                     learning_curve,cross_val_score, train_test_split)
from sklearn.feature_selection import SequentialFeatureSelector as sSFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from scipy.io import savemat
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

def get_first_last_column_names(hrf, df):
    if hrf == "hbo" or hrf == "hbr":
        for column in df.columns:
            if hrf in column:
                first_column = column
                break
        for column in df.loc[:, first_column:].columns:
            if hrf not in column:
                last_column = df.columns[df.columns.get_loc(column) - 1]
                break
    else:
        for column in df.columns:
            if "hbo" in column or "hbr" in column:
                first_column = column
                break
        for column in df.loc[:, first_column:].columns:
            if "hbo" not in column and "hbr" not in column:
                last_column = df.columns[df.columns.get_loc(column) - 1]
                break
    return first_column, last_column


def concat_dfs(features_dict, first_column, last_column):
    # print(features_dict)
    keys = features_dict.keys()
    dataframe_list = list(features_dict.values())
    dataframe_list_2 = []
    for key, dataframe in features_dict.items():
        df = dataframe.loc[:, first_column: last_column]
        df = df.add_suffix(" " + key)
        dataframe_list_2.append(df)

    concat_df = pd.concat(dataframe_list_2, axis="columns")
    concat_df = pd.concat([dataframe_list[0][["ID", "epoch"]], concat_df], axis="columns")

    return concat_df

def add_block_column(df, condition_col: str, group_col: str ):
    df["block"] = df[condition_col].shift() != df[condition_col]
    for subj in df[group_col].unique():
        df.loc[(df["block"] == True) & (df[group_col] == subj), "block"] = list(
            range(1, len(df.loc[(df["block"] == True) & (df[group_col] == subj), "block"]) + 1))

    df.loc[(df["block"] == False), "block"] = np.nan
    df["block"] = df["block"].fillna(method="ffill")
    return df

def feature_extraction(df, hrf: str, group: str, subj_col: str, features: List[str], label: str):
    """ extract features from csv file 
    Parameters:
        path (str): filepath and filename of csv data
        hrf (str): either 'hbo' or 'hbr' or 'hbo+hbr'
        group (str): group to combine e.g. "epoch"
        features (List[str]): features to calculate as List
            possible values: ["max", "min", "peak2peak", "average", "time2max", "slope"]
        label (str): name of the column where the labels are saved
    
    Return:
        labels (series): series of labels for the features
        concat_df (df): dataframe with all features combined; first and second column is ID and epoch.
        features_dict (dict): features with keys specified by the ionput "features" 
            dict values are panda dataframes with "original" as input dataframe
    """
    # create empty dict for output
    # features.append("original")
    features_dict = dict.fromkeys(features)
    df = df.loc[df['time'] >= 0]
    df = df.reset_index(drop = True)
    print(df['condition'].value_counts())
    # delete unnecessary columns from df
    if hrf == "hbo":
        for column in df.columns:
            if "hbr" in column:
                df = df.drop(column, axis=1)
    elif hrf == "hbr":
        for column in df.columns:
            if "hbo" in column:
                df = df.drop(column, axis=1)

    # write original df to output dict
    # features_dict["original"] = df
    # get first and last column for which feature extraction apply
    first_column, last_column = get_first_last_column_names(hrf, df)
    # group df by ID and desired group
    grouped_df = df.groupby([subj_col, group], as_index=False)

    # set labels (y data) for further machine learning
    df_temp = grouped_df.max()
    labels = df_temp[label]

    if "average" in features:
        df_mean = grouped_df.mean(numeric_only=True)
        df_mean = df_mean.drop("time", axis=1)
        features_dict["average"] = df_mean
    if "max" in features or "peak2peak" in features:
        df_max = grouped_df.max()
        df_max = df_max.drop(["time", label], axis=1)
        if "max" in features:
            features_dict["max"] = df_max
    if "min" in features or "peak2peak" in features:
        df_min = grouped_df.min()
        df_min = df_min.drop(["time", label], axis=1)
        if "min" in features:
            features_dict["min"] = df_min
    # peak2 peak
    if "peak2peak" in features:
        df_p2p = grouped_df.max()  # spaceholder for peak to peak
        for column in df.loc[:, first_column:last_column].columns:
            df_p2p[column] = df_max[column].sub(df_min[column])
        df_p2p = df_p2p.drop(["time", label], axis=1)
        features_dict["peak2peak"] = df_p2p
    # time2max
    if "time2max" in features:
        df_idmax = grouped_df.idxmax()
        df_time2max = grouped_df.max()  # placeholder for time to max
        for column in df.loc[:, first_column:last_column].columns:
            for i in range(0, len(df_idmax.index)):
                df_time2max[column].loc[i] = df["time"].iloc[df_idmax[column].iloc[i]]
        df_time2max = df_time2max.drop(["time", label], axis=1)
        features_dict["time2max"] = df_time2max
    # slope
    if "slope" in features:
        # use linear matrix solution of LS problem
        lin_least_sq_sol_list_all = []
        for key in grouped_df.groups:
            current_group = grouped_df.get_group(key)
            time = current_group[["time"]]
            lin_least_sq_sol_list = []
            lin_least_sq_sol_list.append(current_group[group].iloc[0])
            intermediate_solution = np.linalg.pinv(time.T.dot(time)).dot(time.T)
            for column in current_group.loc[:, first_column:last_column].columns:
                lin_least_sq_sol = intermediate_solution.dot(current_group[column].fillna(0))
                lin_least_sq_sol_list.append(float(lin_least_sq_sol))
            lin_least_sq_sol_list.append(current_group[subj_col].iloc[0])

            lin_least_sq_sol_list_all.append(lin_least_sq_sol_list)

        df_lin_ls = pd.DataFrame(lin_least_sq_sol_list_all, columns=current_group.columns[2:])
        # cols = list(df.columns)
        # cols = [cols[-1]] + cols[:-1]
        # df_lin_ls = df_lin_ls[cols]
        features_dict["slope"] = df_lin_ls

    concat_df = concat_dfs(features_dict, first_column, last_column)

    return labels, concat_df, features_dict

# =============================================================================
# Get Pipeline Parameters
# =============================================================================
def get_dict_classifier_GridSearch(SEED, n_jobs):
    classifiers = [
        (LDA(), {'lineardiscriminantanalysis__shrinkage' : ['auto'],
            'lineardiscriminantanalysis__solver' : ['lsqr','eigen']
        }),
        (LogisticRegression(random_state=SEED), {
            'logisticregression__solver': ['liblinear'],
            'logisticregression__multi_class': ['auto'],
            'logisticregression__max_iter': [1000, 3000, 5000],
            'logisticregression__C': [0.01, 0.1, 1, 10, 100],
            'logisticregression__penalty': ['l1', 'l2']
        }),
        (SVC(random_state=SEED), {
            'svc__kernel' : ['linear'],
            'svc__C': [0.01, 0.1, 1, 10, 100]
        }),
        (XGBClassifier(n_jobs=n_jobs, random_state = SEED), dict(xgbclassifier__learning_rate=[0.01, 0.1, 0.3, 0.5],
                          xgbclassifier__max_depth=[3, 4, 5],
                          xgbclassifier__subsample=[0, 0.3, 0.5],
                          xgbclassifier__colsample_bytree=[0.3],
                          xgbclassifier__n_estimators=[5, 7, 10],
                          xgbclassifier__objective=['binary:logistic'])
         ),

        (DummyClassifier(strategy='stratified', random_state=SEED), {})
    ]
    clf_names = ['LDA', 'LR', 'SVM', 'xgBoost', 'Dummy']
    return dict(zip(clf_names, classifiers))
def get_pipeline_settings():
    pipeline_settings = { 'topographical_analysis_interaction_hbo':{'group_interval' : 'epoch',
                                        'contrast': ['HighNeg','LowNeg', 'HighPos',  'LowPos'],
                                        'fe': ["max", "peak2peak", "average"],
                                        'hrf': 'hbo',
                                        'CV_outer': {'splits': 5, 'repeats': 20},
                                        'CV_inner': {'splits': 5, 'shuffle': True},
                                        'scoring':  'f1_weighted',
                                        'SFS': {'k_features': (5,20),  'sfs_forward': True, 'perform': True},
                                        'SEED' : 42, 'n_jobs' : 4},
                          'linear_model_k_best': {'group_interval': 'epoch',
                                               'contrast':  ['HighNeg','LowNeg', 'HighPos','LowPos'],
                                               'fe': ["max", "peak2peak", "average"],
                                               'hrf': 'hbo+hbr',
                                               'CV_outer': {'splits': 5, 'repeats': 20},
                                               'CV_inner': {'splits': 5, 'shuffle': True},
                                               'scoring': 'f1_weighted',
                                               'SFS': {'k_features': (5,20), 'sfs_forward': True, 'perform': True},
                                               'sSFS': {'n_features_to_select': 'auto', 'direction': 'forward', 'scoring': 'f1_weighted', 'n_jobs':7},
                                               'SEED': 42, 'n_jobs': 7},
                        'topographical_analysis_interaction_hbr': {'group_interval': 'epoch',
                                      'contrast': ['HighNeg', 'LowNeg', 'HighPos', 'LowPos'],
                                      'fe': ["max", "peak2peak", "average"],
                                      'hrf': 'hbr',
                                      'CV_outer': {'splits': 5, 'repeats': 20},
                                      'CV_inner': {'splits': 5, 'shuffle': True},
                                      'scoring': 'f1_weighted',
                                      'SFS': {'k_features': (5,20), 'sfs_forward': True, 'perform': True},
                                      'SEED': 42, 'n_jobs': 4}}
    return pipeline_settings
def get_roi_integer(roi):
    roi_integer = dict(zip(list(roi.keys()), [[]] * len(roi.keys())))
    for r in roi:
        for ch in roi[r]:
            full_ch = ch.replace('S', '').replace('D', '').split('_')
            roi_integer[r] = roi_integer[r] + [[int(full_ch[0]), int(full_ch[1])]]
    return roi_integer

def categorize_brodmann(ba_desc):
    ba_desc_lower = ba_desc.lower()  # Convert to lower case
    if "dorsolateral prefrontal cortex" in ba_desc_lower:
        return ba_desc_lower[0] + "dlPFC"
    elif "pre-motor" in ba_desc_lower or "supplementary motor cortex" in ba_desc_lower:
        return "premotor"
    elif "pars triangularis" in ba_desc_lower or "pars opercularis" in ba_desc_lower or 'inferior prefrontal gyrus' in ba_desc_lower:
        return ba_desc_lower[0] + "IFG"
    elif 'orbitofrontal' in ba_desc_lower:
        return ba_desc_lower[0] + "OFC"
    elif "temporopolar area" in ba_desc_lower:
        return "fronto-temporal"
    return None


def get_roi_ch_mappings(project):
    if 'nircademy' in project:
        roi = {
            "rdlPFC": ['S6_D5', 'S6_D6', 'S6_D7',
                       'S7_D5', 'S7_D7', 'S7_D9',
                       'S8_D6', 'S8_D7',
                       'S11_D5', 'S11_D9'],
            "ldlPFC": ['S1_D1', 'S1_D2', 'S1_D13',
                       'S2_D1', 'S2_D3',
                       'S3_D1', 'S3_D2', 'S3_D3',
                       'S13_D2', 'S13_D13'],
            "mid-dlPFC": ['S3_D4', 'S4_D2', 'S4_D4', 'S4_D5', 'S4_D14',
                          'S5_D3', 'S5_D4', 'S5_D6', 'S6_D4'],
            "premotor": ['S9_D7', 'S9_D8', 'S9_D9',
                         'S10_D8', 'S10_D9', 'S10_D10', 'S11_D10',
                         'S12_D10', 'S12_D12', 'S12_D14',
                         'S14_D1', 'S14_D11', 'S14_D13',
                         'S15_D11', 'S15_D12', 'S15_D13', 'S11_D14', 'S13_D12', 'S13_D14']
        }

        channel_mappings = {
        "S1": "F3",
        "S2": "AF7",
        "S3": "AF3",
        "S4": "Fz",
        "S5": "Fpz",
        "S6": "AF4",
        "S7": "F4",
        "S8": "AF8",
        "S9": "FC6",
        "S10": "C4",
        "S11": "FC2",
        "S12": "Cz",
        "S13": "FC1",
        "S14": "FC5",
        "S15": "C3",
        "D1": "F5",
        "D2": "F1",
        "D3": "Fp1",
        "D4": "AFz",
        "D5": "F2",
        "D6": "Fp2",
        "D7": "F6",
        "D8": "C6",
        "D9": "FC4",
        "D10": "C2",
        "D11": "C5",
        "D12": "C1",
        "D13": "FC3",
        "D14": "FCz"
        }
    if 'mikado' in project:
        pos_mapping = pd.read_csv(os.path.join('toolbox/ch_pos_mapping.csv'), header = 0, sep = ';', decimal =',').ffill()
        # Simplify Source and Detector by removing additional details
        pos_mapping['EEG_CH_S'] = pos_mapping['Source'].apply(lambda x: x.split('(')[-1].strip(')'))
        pos_mapping['EEG_CH_D'] = pos_mapping['Detector'].apply(lambda x: x.split('(')[-1].strip(')'))
        pos_mapping['Source'] = pos_mapping['Source'].apply(lambda x: x.split(' ')[0])
        pos_mapping['Detector'] = pos_mapping['Detector'].apply(lambda x: x.split(' ')[0])
        channel_mappings = {row['Source']: row['EEG_CH_S'] for index, row in pos_mapping.iterrows()}
        channel_mappings.update({row['Detector']: row['EEG_CH_D'] for index, row in pos_mapping.iterrows()})
        roi = {}
        for index, row in pos_mapping.iterrows():
            category = categorize_brodmann(row['Brodmann Area'])
            if category:
                key = f"{row['Source']}_{row['Detector']}"
                if category not in roi:
                    roi[category] = []
                roi[category].append(key)
            else:
                print(f"Label {row['Brodmann Area']} does not fit in the predefined ROI labels.")

        print("ROI:")
        print(roi)

    else:
        print('No mapping specified')
        roi = None
        channel_mappings = None
    roi_integer = get_roi_integer(roi)
    return roi, roi_integer, channel_mappings

def get_ROI_dlPFC(project):
    # Specify channel pairs for each ROI
    roi, roi_integer, channel_mappings = get_roi_ch_mappings(project)
    rdlPFC = roi_integer['rdlPFC']
    ldlPFC = roi_integer['ldlPFC']
    names_rdlPFC = []
    names_ldlPFC = []
    for pair_right, pair_left in zip(rdlPFC, ldlPFC):
        names_rdlPFC.append('S' + str(pair_right[0]) + '_D' + str(pair_right[1]) + ' hbo')
        names_rdlPFC.append('S' + str(pair_right[0]) + '_D' + str(pair_right[1]) + ' hbr')
        names_ldlPFC.append('S' + str(pair_left[0]) + '_D' + str(pair_left[1]) + ' hbo')
        names_ldlPFC.append('S' + str(pair_left[0]) + '_D' + str(pair_left[1]) + ' hbr')
    return names_rdlPFC, names_ldlPFC
# =============================================================================
# Single time Decoding with different Feature Extraction Methods, Linear Models
# and nested cross-validation per subject
# combine cross_val with Grid_Search_CV
# =============================================================================
def single_trial_decoding_linear_models(subj, X, y, classifier, pipeline_settings, dict_gs, save):
    unique, count = np.unique(y, return_counts=True)
    dict_subj_counts = dict(zip(list(unique), list(count)))
    dict_subj_counts['subj'] = subj
    count_subj = pd.DataFrame(dict_subj_counts, index=[subj])
    if np.all(count <= 30):
        print(subj, count, 'TOO LESS SAMPLES FOR DECODING')
    # =============================================================================
    # Define contrast name
    # =============================================================================
    if len(pipeline_settings['contrast']) == 2:
        a_vs_b = '%s_vs_%s' % (os.path.basename(pipeline_settings['contrast'][0]),
                           os.path.basename(pipeline_settings['contrast'][1]))
        # =============================================================================
        # Encode y
        # =============================================================================

        y = y.replace({pipeline_settings['contrast'][0]: 0, pipeline_settings['contrast'][1]: 1}).values
        y = LabelEncoder().fit_transform(y)
    elif len(pipeline_settings['contrast']) == 3:
        a_vs_b = '%s_vs_%s_vs_%s' % (os.path.basename(pipeline_settings['contrast'][0]),
                           os.path.basename(pipeline_settings['contrast'][1]),
                           os.path.basename(pipeline_settings['contrast'][2]))
        # =============================================================================
        # Encode y
        # =============================================================================

        y = y.replace({pipeline_settings['contrast'][0]: 0, pipeline_settings['contrast'][1]: 1, pipeline_settings['contrast'][2]: 2}).values
        # y = LabelEncoder().fit_transform(y)
    elif len(pipeline_settings['contrast']) == 4:
        a_vs_b = '%s_vs_%s_vs_%s_vs_%s' % (os.path.basename(pipeline_settings['contrast'][0]),
                           os.path.basename(pipeline_settings['contrast'][1]),
                           os.path.basename(pipeline_settings['contrast'][2]),
                           os.path.basename(pipeline_settings['contrast'][3]))
        # =============================================================================
        # Encode y
        # =============================================================================

        y = y.replace({pipeline_settings['contrast'][0]: 0, pipeline_settings['contrast'][1]: 1, pipeline_settings['contrast'][2]: 2, pipeline_settings['contrast'][3]: 3}).values
        y = LabelEncoder().fit_transform(y)
    # =============================================================================
    # ML PIPELINE
    # =============================================================================
    # make pipeline for classification

    clf_pipe = make_pipeline(
        StandardScaler(),
        dict_gs[0])
    # instantiate the SK-Fold for the GridSearch CV hyperparamerter optimization
    cv_inner = StratifiedKFold(n_splits=pipeline_settings['CV_inner']['splits'], random_state=pipeline_settings['SEED'],
                               shuffle=pipeline_settings['CV_inner']['shuffle'])

    # instantiate the GridSearchCV for parameter optimization - use the clf pipeline
    gs = GridSearchCV(clf_pipe, param_grid=dict_gs[1],
                      scoring=pipeline_settings['scoring'],
                      cv=cv_inner,
                      n_jobs=pipeline_settings['n_jobs'],
                      refit=True,
                      return_train_score=True)

    cv_outer = RepeatedStratifiedKFold(n_splits=pipeline_settings['CV_outer']['splits'],
                                       random_state=pipeline_settings['SEED'],
                                       n_repeats=pipeline_settings['CV_outer']['repeats'])

    if (classifier != 'xgBoost') and ('topographical_analysis' not in save) and (pipeline_settings['SFS']['perform']):
        # Feature Selection
        sfs = SFS(gs, k_features=pipeline_settings['SFS']['k_features'],
                  forward=pipeline_settings['SFS']['sfs_forward'], floating=False,
                  scoring=pipeline_settings['scoring'],
                  cv=cv_outer, n_jobs=pipeline_settings['n_jobs'], verbose=2)
        # sfs = sSFS(gs, **pipeline_settings['sSFS'], cv=cv_outer)

        sfs = sfs.fit(X, y)
        # Save results
        df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
        df.to_csv("{}/scores_with_SFS_{}_{}.csv".format(save, subj, a_vs_b), sep=";", decimal=',', header=True,
                  index=False)

        # Plots
        for kind in ['std_dev', 'std_err']:
            fig = plot_sfs(sfs.get_metric_dict(), kind=kind)
            plt.ylim([0.0, 1])
            plt.title('{} {}-Class Sequential Forward Selection (w. {})'.format(classifier,
                                                                                len(pipeline_settings['contrast']),
                                                                                kind))
            plt.grid()
            plt.tight_layout
            plt.savefig("{}/sfs_{}_{}_{}.png".format(save, subj, a_vs_b, kind))
            plt.clf()

    # execute the nested cross-validation - do the nested cross-validation on the whole dataset
    # the return is a dict
    scores = cross_validate(gs, X, y, scoring=pipeline_settings['scoring'],
                            cv=cv_outer, n_jobs=pipeline_settings['n_jobs'],
                            return_train_score=True, error_score='raise')
    if (classifier not in ['Dummy', 'xgBoost']):
        coefficients = gs.fit(X, y).best_estimator_.steps[-1][1].coef_

    if (classifier in ['LR', 'LDA']):
        patterns = list()
        for train, test in cv_outer.split(X, y):

            pattern = get_coef(make_pipeline(StandardScaler(), LinearModel(dict_gs[0])).fit(np.array(X)[train], y[train]), attr='patterns_', inverse_transform=True)
            patterns.append(pattern)
        patterns = np.mean(np.array(patterns), axis=0)
    else:
	    patterns = np.nan

    if (classifier == 'xgBoost'):
        coefficients = gs.fit(X, y).best_estimator_.steps[-1][1].feature_importances_

    if (classifier == 'Dummy'):
        coefficients = np.nan


    print('Cross-validation on the test set (' + classifier + ' Clf) = %1.3f '
                                                                      '(+/- %1.3f)' % (
              np.nanmean(scores['test_score']), np.nanstd(scores['test_score'])))

    print('Cross-validation on the train set (' + classifier + ' Clf) = %1.3f '
                                                                       '(+/- %1.3f)' % (
              np.nanmean(scores['train_score']), np.nanstd(scores['train_score'])))

    fname_clf = os.path.join(save, 'scores_without_SFS_{}_{}.mat'.format(subj, a_vs_b))
    savemat(fname_clf, {'scores': scores})
    df_scores = pd.DataFrame(data=scores)
    df_scores.to_csv("{}/scores_without_SFS_{}_{}.csv".format(save, subj, a_vs_b), sep=";", decimal=',', header=True,
                     index=False)

    fname_coef = os.path.join(save, 'coefficients_without_SFS_{}_{}.mat'.format(subj, a_vs_b))
    savemat(fname_coef, {'coefficients': coefficients, 'features': np.array(list(X.columns))})
    fname_patterns = os.path.join(save, 'patterns_without_SFS_{}_{}.mat'.format(subj, a_vs_b))
    savemat(fname_patterns, {'patterns': patterns, 'features': np.array(list(X.columns))})
    return count_subj

def single_trial_decoding_linear_models_get_patterns(subj, X, y, classifier, pipeline_settings, dict_gs, save):
    if classifier not in ['LR', 'LDA']:
        return

    unique, count = np.unique(y, return_counts=True)
    if np.all(count <= 30):
        print(subj, count, 'TOO LESS SAMPLES FOR DECODING')
    # =============================================================================
    # Define contrast name
    # =============================================================================
    if len(pipeline_settings['contrast']) == 2:
        a_vs_b = '%s_vs_%s' % (os.path.basename(pipeline_settings['contrast'][0]),
                               os.path.basename(pipeline_settings['contrast'][1]))
        # =============================================================================
        # Encode y
        # =============================================================================

        y = y.replace({pipeline_settings['contrast'][0]: 0, pipeline_settings['contrast'][1]: 1}).values
        y = LabelEncoder().fit_transform(y)
    elif len(pipeline_settings['contrast']) == 3:
        a_vs_b = '%s_vs_%s_vs_%s' % (os.path.basename(pipeline_settings['contrast'][0]),
                                     os.path.basename(pipeline_settings['contrast'][1]),
                                     os.path.basename(pipeline_settings['contrast'][2]))
        # =============================================================================
        # Encode y
        # =============================================================================

        y = y.replace({pipeline_settings['contrast'][0]: 0, pipeline_settings['contrast'][1]: 1,
                       pipeline_settings['contrast'][2]: 2}).values
        # y = LabelEncoder().fit_transform(y)
    elif len(pipeline_settings['contrast']) == 4:
        a_vs_b = '%s_vs_%s_vs_%s_vs_%s' % (os.path.basename(pipeline_settings['contrast'][0]),
                                           os.path.basename(pipeline_settings['contrast'][1]),
                                           os.path.basename(pipeline_settings['contrast'][2]),
                                           os.path.basename(pipeline_settings['contrast'][3]))
        # =============================================================================
        # Encode y
        # =============================================================================

        y = y.replace({pipeline_settings['contrast'][0]: 0, pipeline_settings['contrast'][1]: 1,
                       pipeline_settings['contrast'][2]: 2, pipeline_settings['contrast'][3]: 3}).values
        y = LabelEncoder().fit_transform(y)

    cv_outer = RepeatedStratifiedKFold(n_splits=pipeline_settings['CV_outer']['splits'],
                                       random_state=pipeline_settings['SEED'],
                                       n_repeats=pipeline_settings['CV_outer']['repeats'])

    if (classifier in ['LR', 'LDA']):
        patterns = list()
        for train, test in cv_outer.split(X, y):

            pattern = get_coef(make_pipeline(StandardScaler(), LinearModel(dict_gs[0])).fit(np.array(X)[train], y[train]), attr='patterns_', inverse_transform=True)
            patterns.append(pattern)
        patterns = np.mean(np.array(patterns), axis=0)
    fname_patterns = os.path.join(save, 'patterns_without_SFS_{}_{}.mat'.format(subj, a_vs_b))
    savemat(fname_patterns, {'patterns': patterns, 'features': np.array(list(X.columns))})
    return
def create_df_boot_scores(data_path, save_path, analysis, specification, feature, score_list, contrast):
    df_boot = pd.DataFrame()
    for classifier in os.listdir(os.path.join(data_path, analysis, specification, feature)):
        for subj in range(2, 20):
            print(subj)
            for score_file in score_list:
                scores = loadmat(os.path.join(data_path, analysis, specification, feature, classifier, score_file[0] + '_' + str(subj) + '_' + contrast + score_file[1]))
                test = scores['scores'][0][0][2]
                train = scores['scores'][0][0][3]
                df_boot_subj = pd.DataFrame({'subj': len(test.flatten()) * [subj], 'classifier': len(test.flatten()) * [classifier + score_file[2]],
                                             'train': train.flatten(),'test': test.flatten(), 'fold': list(range(1, len(test.flatten()) + 1))})
                df_boot = pd.concat((df_boot, df_boot_subj))
    os.makedirs(os.path.join(save_path, analysis, specification, feature), exist_ok=True)
    df_boot.to_csv(os.path.join(save_path, analysis, specification, feature,'df_boot.csv'), header = True, index = False, decimal=',', sep = ';')
    return df_boot

def create_df_boot_SFS_scores(data_path, save_path, analysis, specification, feature, score_file, contrast):
    df_boot = pd.DataFrame()
    for classifier in os.listdir(os.path.join(data_path, analysis, specification, feature)):
        for subj in range(2, 20):
            list_results = [s for s in os.listdir(os.path.join(data_path, analysis, specification, feature, classifier)) if str(subj) == s.split(score_file[0])[-1].strip('_').split('_')[0]]
            if len(list_results) < 1:
                print(subj)
            scores = pd.read_csv(os.path.join(data_path, analysis, specification, feature, classifier, score_file[0] + '_' + str(subj) + '_' + contrast + score_file[1]), sep=';', decimal = ',', header = 0)
            initial_feature_list = []
            for k in range(0, len(scores)):
                print('initial_feature_list:', initial_feature_list)
                test = np.array([float(numb.replace('\n', '')) for numb in scores['cv_scores'][k].strip('][').split(' ') if len(numb) > 1])
                list_features = np.array([str(string) for string in scores['feature_names'][k].strip('()').split(', ')])
                features = []
                for feat in list_features:
                    features.append("".join(list([val for val in feat
                    if val.isalpha() or val.isnumeric() or (val == ' ') or (val =='_')])))

                print('features:', features)
                for feat in initial_feature_list:
                    if feat not in features:
                        initial_feature_list.remove(feat)
                for feat in features:
                    if feat not in initial_feature_list:
                        initial_feature_list.append(feat)
                print('after comparing features:', initial_feature_list)
                df_boot_subj_k = pd.DataFrame({'subj': len(test.flatten()) * [subj], 'classifier': len(test.flatten()) * [classifier + score_file[2]], 'test': test.flatten(), 'fold': list(range(1, len(test.flatten()) + 1)), 'k' : len(test.flatten()) * [k + 1]})
                assert set(initial_feature_list) == set(features)
                for k_feat in range(0, len(scores)):
                    df_boot_subj_k['k_' + str(k_feat + 1)] = np.nan
                for k_feat in range(0, len(initial_feature_list)):
                    print('add feat', initial_feature_list[k_feat])
                    df_boot_subj_k['k_' + str(k_feat + 1)] = initial_feature_list[k_feat]
                df_boot = df_boot.append(df_boot_subj_k)
    if not os.path.exists(os.path.join(save_path, analysis, specification, feature)):
        os.makedirs(os.path.join(save_path, analysis, specification, feature))
    df_boot.to_csv(os.path.join(save_path, analysis, specification, feature,'df_boot_SFS.csv'), header = True, index = False, decimal=',', sep = ';')
    return df_boot


def create_df_coefficients(data_path, save_path, analysis, specification, feature, contrast):
    df_coef = pd.DataFrame()
    for classifier in [cl for cl in os.listdir(os.path.join(data_path, analysis, specification, feature)) if not '.csv' in cl]:
        for subj in range(2, 20):
            print(subj)
            if classifier != 'Dummy':
                coef = loadmat(os.path.join(data_path, analysis, specification, feature, classifier, 'coefficients_without_SFS_' + str(subj) + '_' + contrast + '.mat'))
                for con in range(0, coef['coefficients'].shape[0]):
                    con_name = contrast.split('_vs_')[con]
                    df_coef_subj = pd.DataFrame({'subj': len(coef['coefficients'][con].flatten()) * [subj],
                                                 'con' : len(coef['coefficients'][con].flatten()) * [con_name],
                                                 'classifier': len(coef['coefficients'][con].flatten()) * [classifier],
                                                 'coef': coef['coefficients'][con].flatten(),
                                                 'features': coef['features']})
                    df_coef = pd.concat((df_coef, df_coef_subj))
    os.makedirs(os.path.join(save_path, analysis, specification, feature), exist_ok=True)
    df_coef.to_csv(os.path.join(save_path, analysis, specification, feature,'df_coef.csv'), header = True, index = False, decimal=',', sep = ';')
    return df_coef

def create_df_patterns(data_path, save_path, analysis, specification, feature, contrast):
    df_patterns = pd.DataFrame()
    for classifier in [cl for cl in os.listdir(os.path.join(data_path, analysis, specification, feature)) if not '.csv' in cl]:
        for subj in range(2, 20):
            print(subj)
            if classifier != 'Dummy':
                try:
                    patterns = loadmat(os.path.join(data_path, analysis, specification, feature, classifier, 'patterns_without_SFS_' + str(subj) + '_' + contrast + '.mat'))
                except:
                    print('No File for ', subj, 'analysis', os.path.split(data_path)[-1])
                    continue
                for con in range(0, patterns['patterns'].shape[0]):
                    con_name = contrast.split('_vs_')[con]
                    df_patterns_subj = pd.DataFrame({'subj': len(patterns['patterns'][con].flatten()) * [subj],
                                                 'con' : len(patterns['patterns'][con].flatten()) * [con_name],
                                                 'classifier': len(patterns['patterns'][con].flatten()) * [classifier],
                                                 'patterns': patterns['patterns'][con].flatten(),
                                                 'features': patterns['features']})
                    df_patterns = pd.concat((df_patterns, df_patterns_subj))

    os.makedirs(os.path.join(save_path, analysis, specification, feature), exist_ok=True)
    df_patterns.to_csv(os.path.join(save_path, analysis, specification, feature,'df_patterns.csv'), header = True, index = False, decimal=',', sep = ';')
    return df_patterns


def count_features(df_best_k, best_k, roi):
    counts = {
        "raw_feat": {},  # As it is in the data
        "hbr_hbo": {"hbr": 0, "hbo": 0},  # Only HbR and HbO
        "channel": {},  # Only Channels S-D
        "val_feat": {},  # P2P, min, max, Slope etc.
        "hbr_hbo_val": {},  # Combination of hbr/hbo and val_feat
        "roi": {},  # Regions of interest (l-dlPFC, r-dlPFC, mid-dlPFC, pre-motor)
        "roi_hbr_hbo_val": {}  # Combination of ROI, hbr/hbo and val feat
    }

    actual_channels = {
        "ldlPFC": [],
        "rdlPFC": [],
        "mid-dlPFC": [],
        "premotor": [],
        "others" : [],
    }

    for subj in [s for s in df_best_k['subj'].unique() if s != 'average']:
        df_sub = df_best_k.loc[df_best_k["subj"] == subj]

        for k in range(1, best_k + 1):
            feat = df_sub.iloc[0][f"k_{k}"]
            comp = feat.split(" ")
            # Raw feature count
            if feat in counts['raw_feat']:
                counts['raw_feat'][feat] += 1
            else:
                counts['raw_feat'][feat] = 1

            #  hbr and hbo feature count
            if "hbr" in feat:
                counts['hbr_hbo']["hbr"] += 1
            elif "hbo" in feat:
                counts['hbr_hbo']["hbo"] += 1

            # Channel count
            if comp[0] in counts['channel']:
                counts['channel'][comp[0]] += 1
            else:
                counts['channel'][comp[0]] = 1

            # Value Features (min, max, Peak2Peak etc.)
            if comp[2] in counts['val_feat']:
                counts['val_feat'][comp[2]] += 1
            else:
                counts['val_feat'][comp[2]] = 1

            # Combination from value features and hbr/hbo
            key_ = f"{comp[1]} {comp[2]}"
            if key_ in counts['hbr_hbo_val']:
                counts['hbr_hbo_val'][key_] += 1
            else:
                counts['hbr_hbo_val'][key_] = 1

            # Regions of interest
            roi_label = "others"
            for key in roi:
                if comp[0] in roi[key]:
                    roi_label = key
                    break
            if roi_label in counts['roi']:
                counts['roi'][roi_label] += 1
            else:
                counts['roi'][roi_label] = 1

            # Add channel to the actual channel list (to look which channels are actual available in the dataset)
            if comp[0] not in actual_channels[roi_label]:
                actual_channels[roi_label].append(comp[0])

            # Roi + hbr/hbo + val feats
            key_ = f"{roi_label} {comp[1]} {comp[2]}"
            if key_ in counts['roi_hbr_hbo_val']:
                counts['roi_hbr_hbo_val'][key_] += 1
            else:
                counts['roi_hbr_hbo_val'][key_] = 1

    counts_ch_per_feat_chroma = {}
    channels = [f.split(' ')[0] for f in sorted(list(counts['raw_feat'].keys()))]
    chromosome = [f.split(' ')[1] for f in sorted(list(counts['raw_feat'].keys()))]
    feat = [f.split(' ')[2] for f in sorted(list(counts['raw_feat'].keys()))]
    for i, f in enumerate(feat):
        if chromosome[i] + '_' + feat[i] not in list(counts_ch_per_feat_chroma.keys()):
            counts_ch_per_feat_chroma[chromosome[i] + '_' + feat[i]] = [channels[i]]
        else:
            counts_ch_per_feat_chroma[chromosome[i] + '_' + feat[i]].append(channels[i])

    return counts, actual_channels, counts_ch_per_feat_chroma
