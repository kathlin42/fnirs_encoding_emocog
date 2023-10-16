# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 18:42:07 2021

@author: hirning
"""
# =============================================================================
# Import required packages
# =============================================================================
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

mne.viz.set_3d_backend("pyvista")  # pyvistaqt
import mne_nirs
import statsmodels.formula.api as smf
from natsort import natsort_keygen
from toolbox import config_analysis
# =============================================================================
# Paths and Variables
# =============================================================================
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'~qgis directory\apps\Qt5\plugins'
os.environ['PATH'] += r';~qgis directory\apps\qgis\bin;~qgis directory\apps\Qt5\bin'

analysis_settings = 'fNIRS_GLM_window_'
include_silence = ''
handedness = '' #with_handedness_
data_directory = os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_preproc', handedness + analysis_settings + str(config_analysis.GLM_time_window) + include_silence)
save_directory = os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_encoding', handedness + analysis_settings + str(config_analysis.GLM_time_window) + include_silence)

if not os.path.exists("{}".format(save_directory)):
    print('creating path for saving')
    os.makedirs("{}".format(save_directory))

cmap = mpl.cm.seismic
# =============================================================================
# Load Data
# =============================================================================
conditions = ['LowNeu', 'LowPos', 'LowNeg', 'HighNeu', 'HighPos', 'HighNeg']
# =============================================================================
# Plot LMM Results on 3D fsaverage Brain
# =============================================================================
plots = ['contrast_inter_EMO_NegPos', 'contrast_inter_EMO_NeuNeg', 'contrast_inter_EMO_NeuPos',
         #'contrast_inter_WL_NegPos', 'contrast_inter_WL_NeuNeg', 'contrast_inter_WL_NeuPos',
         'main_emo_NegPos', 'main_emo_NeuNeg', 'main_emo_NeuPos',
         'main_wl_HighLow']#,

         #'contrast_low_NegPos', 'contrast_low_NeuNeg', 'contrast_low_NeuPos',
         #'contrast_high_NegPos', 'contrast_high_NeuNeg', 'contrast_high_NeuPos',
         #'contrast_neg_HighLow', 'contrast_neu_HighLow', 'contrast_pos_HighLow']
if 'silence' in include_silence:
    plots = plots + ['main_wl_task_HighLow', 'main_wl_dis_HighLow']
df_cons = pd.DataFrame()
# Cortical Surface Projections for Contrasts
for pick in ['hbo', 'hbr']:
    # pick = 'hbo'
    for cidx, contrast in enumerate(plots):

        # Replace Values with Coefficients calculated in R lmer
        df_con_coef = pd.read_csv(os.path.join(data_directory, 'LMM_coefficients', 'coefficients_' + handedness + contrast + '_' + pick + '.csv'))
        df_con_coef = df_con_coef.loc[df_con_coef["Unnamed: 0"] == 'handright']
        df_con_coef['Significant'] = [False if np.min(np.linspace(row_lower, row_upper, 50)) < 0 < np.max(np.linspace(row_lower, row_upper, 50)) else True for row_lower, row_upper in zip(df_con_coef['[0.025'], df_con_coef['0.975]'])]
        df_con_coef['Contrast'] = contrast
        df_con_coef['Chromophore'] = pick
        df_cons = df_cons.append(df_con_coef)


df_cons['95%CI'] = [ '[' + str(np.round(lower, 3)) + '; ' + str(np.round(upper, 3)) + ']' for lower, upper in zip(df_cons['[0.025'], df_cons['0.975]'])]
df_cons.drop(['Unnamed: 0', '[0.025', '0.975]', 'Significant'], axis = 1, inplace = True)
neworder = ['Contrast', 'Chromophore', 'Estimate', 'Std. Error', 'df', 't value', 'Pr(>|t|)', '95%CI']
df_cons = df_cons.reindex(columns=neworder)
df_cons['Contrast'] = df_cons['Contrast'].replace(dict(zip(df_cons['Contrast'].unique(), ['Interaction Negative - Positive High - Low',
                                                                    'Interaction Neutral - Negative High - Low',
                                                                    'Interaction Neutral - Positive High - Low',
                                                                    'Main Emotion Negative - Positive',
                                                                    'Main Emotion Neutral - Negative',
                                                                    'Main Emotion Neutral - Positive',
                                                                    'Main Workload High - Low'])))
for col in ['Estimate', 'Std. Error', 't value', 'Pr(>|t|)']:
    df_cons[col] = [str(np.round(val, 3)) for val in df_cons[col]]
df_cons['df'] = 16

df_reanalyze = pd.read_csv(os.path.join(os.path.split(data_directory)[0], 'REANALYZE_S9_D11_hbr_coefficients_contrast_inter_EMO_NegPos.csv'))

#%%
from scipy.stats import spearmanr
ch_data = pd.read_csv(os.path.join(data_directory, 'nirs_glm_cha.csv'), sep = ';', decimal = ',')
ch_data['Emotion_Condition'] = [val[-3:] for val in ch_data['Condition']]
ch_data['Load_Condition'] = [val[:-3] for val in ch_data['Condition']]
ch_data = ch_data.loc[(ch_data['ch_name'] == 'S9_D11 hbr') & (ch_data['Emotion_Condition'].isin(['Pos', 'Neg']))]
beh_data = pd.read_csv(os.path.join(config_analysis.project_directory, 'sourcedata/performance/performance_aggregated.csv'), sep = ';', decimal = ',')
beh_data = beh_data.loc[:, ['Subject', 'Condition', 'Accuracy', 'Speed']]
for col in ['Accuracy', 'Speed']:
    beh_data[col] = (beh_data[col] - beh_data[col].min())/(beh_data[col].max() - beh_data[col].min())
data = ch_data.merge(beh_data, left_on=['ID', 'Condition'], right_on = ['Subject', 'Condition'])
results = pd.DataFrame()
for performance_measure in ['Accuracy', 'Speed']:
    for con in data['Condition'].unique():
        theta = data.loc[data['Condition'] == con, 'theta']
        accuracy = data.loc[data['Condition'] == con, performance_measure]
        stats, p = spearmanr(theta, accuracy)
        results = results.append(pd.DataFrame({'Performance_measure': [performance_measure],
                                               'Condition': [con],
                                               'stats': [stats],
                                               'p': [p]}))