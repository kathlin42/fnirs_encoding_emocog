# -*- coding: utf-8 -*-
# =============================================================================
# Directories and Imports
# =============================================================================
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from toolbox import config_analysis, helper_plot
# =============================================================================
# Set Variables for Plotting
# =============================================================================
analysis_settings = 'fNIRS_GLM_window_'
include_silence = '_correct_silence'#'_include_silence'
analysis_settings = 'fNIRS_GLM_window_'
only_silence = True
data_directory = os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_preproc', analysis_settings + str(config_analysis.GLM_time_window) + include_silence)
save_directory = os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_encoding', analysis_settings + str(config_analysis.GLM_time_window) + include_silence, 'stats_chs')

# Contrasts
plots = ['contrast_pos_HighLow', 'contrast_neu_HighLow', 'contrast_neg_HighLow',
         'contrast_low_NegPos', 'contrast_low_NeuNeg', 'contrast_low_NeuPos',
         'contrast_high_NegPos', 'contrast_high_NeuNeg', 'contrast_high_NeuPos',

         'contrast_inter_EMO_NegPos', 'contrast_inter_EMO_NeuNeg', 'contrast_inter_EMO_NeuPos',
         'contrast_inter_WL_NegPos', 'contrast_inter_WL_NeuNeg', 'contrast_inter_WL_NeuPos',

         'main_emo_NegPos', 'main_emo_NeuNeg', 'main_emo_NeuPos',
         'main_wl_HighLow']
sub_conditions = [['HighPos', 'LowPos'], ['HighNeu', 'LowNeu'], ['HighNeg', 'LowNeg'],
                  ['LowNeg', 'LowPos'], ['LowNeu', 'LowNeg'], ['LowNeu', 'LowPos'],
                  ['HighNeg', 'HighPos'], ['HighNeu', 'HighNeg'], ['HighNeu', 'HighPos'],

                  ['HighNeg', 'HighPos', 'LowNeg', 'LowPos'], ['HighNeu', 'HighNeg', 'LowNeu', 'LowNeg'],
                  ['HighNeu', 'HighPos', 'LowNeu', 'LowPos'],
                  ['HighNeg', 'LowNeg', 'HighPos', 'LowPos'], ['HighNeu', 'LowNeu', 'HighNeg', 'LowNeg'],
                  ['HighNeu', 'LowNeu', 'HighPos', 'LowPos'],

                  ['HighNeg', 'LowNeg', 'HighPos', 'LowPos'], ['HighNeu', 'LowNeu', 'HighNeg', 'LowNeg'],
                  ['HighNeu', 'LowNeu', 'HighPos', 'LowPos'],
                  ['HighNeg', 'HighNeu', 'HighPos', 'LowNeg', 'LowNeu', 'LowPos']]

if 'include_silence' in include_silence:
    if only_silence:
        plots = ['main_wl_HighLow']
        sub_conditions = [['HighNeg', 'HighNeu', 'HighPos', 'HighSil', 'LowNeg', 'LowNeu', 'LowPos', 'LowSil']]
    plots = plots + ['main_wl_task_HighLow', 'main_wl_dis_HighLow',
                     'main_emo_NegSil', 'main_emo_NeuSil', 'main_emo_PosSil',
                     'contrast_low_NegSil', 'contrast_low_NeuSil','contrast_low_PosSil',
                     'contrast_high_NegSil', 'contrast_high_NeuSil','contrast_high_PosSil',
                     'contrast_inter_WL_NeuSil', 'contrast_inter_WL_NegSil', 'contrast_inter_WL_PosSil',
                     'contrast_inter_EMO_NeuSil', 'contrast_inter_EMO_NegSil', 'contrast_inter_EMO_PosSil']

    sub_conditions = sub_conditions + [['HighSil', 'LowSil'], ['HighNeg', 'HighNeu', 'HighPos', 'HighSil', 'LowNeg', 'LowNeu', 'LowPos', 'LowSil'],
                                       ['HighNeg', 'HighSil', 'LowNeg', 'LowSil'], ['HighNeu', 'HighSil', 'LowNeu', 'LowSil'], ['HighPos', 'HighSil', 'LowPos', 'LowSil'],
                                       ['LowNeg', 'LowSil'], ['LowNeu', 'LowSil'], ['LowPos', 'LowSil'],
                                       ['HighNeg', 'HighSil'], ['HighNeu', 'HighSil'], ['HighPos', 'HighSil'],
                                       ['HighNeu', 'LowNeu', 'HighSil', 'LowSil'], ['HighNeg', 'LowNeg', 'HighSil', 'LowSil'],
                                       ['HighPos', 'LowPos', 'HighSil', 'LowSil'],
                                       ['HighNeu', 'HighSil', 'LowNeu', 'LowSil'], ['HighNeg','HighSil', 'LowNeg', 'LowSil'],
                                       ['HighPos', 'HighSil', 'LowPos', 'LowSil']]
elif 'correct_silence' in include_silence:
    if only_silence:
        plots = ['main_wl_HighLow']
        sub_conditions = [['HighNeg', 'HighNeu', 'HighPos', 'HighSil', 'LowNeg', 'LowNeu', 'LowPos', 'LowSil']]
    plots = plots + ['main_wl_task_HighLow', 'main_wl_dis_HighLow']
    sub_conditions = sub_conditions + [['HighSil', 'LowSil'], ['HighNeg', 'HighNeu', 'HighPos', 'HighSil', 'LowNeg', 'LowNeu', 'LowPos', 'LowSil']]

dict_contrast_subcondition = dict(zip(plots, sub_conditions))
# =============================================================================
# Load Data
# =============================================================================
df_cha = pd.read_csv(os.path.join(data_directory, 'nirs_glm_cha.csv'), sep=';', decimal=',')

for pick in ['hbo', 'hbr']:
    for cidx, contrast in enumerate(plots):
        # contrast = 'main_wl_HighLow'
        # Replace Values with Coefficients calculated in R lmer

        df_con_coef = pd.read_csv(os.path.join(data_directory, 'LMM_coefficients', 'coefficients_' + contrast + '_' + pick + '.csv'))
        df_con_coef['Significant'] = [False if np.min(np.linspace(row_lower, row_upper, 50)) < 0 < np.max(np.linspace(row_lower, row_upper, 50)) else True for row_lower, row_upper in zip(df_con_coef['[0.025'], df_con_coef['0.975]'])]
        print(df_con_coef.loc[df_con_coef['Significant'] == True, ['[0.025', '0.975]']])
        plotting_ch = df_con_coef.loc[df_con_coef['Significant'] == True]
        for i_ch, ch in plotting_ch.iterrows():
            print(ch['Unnamed: 0'][7:])
            df_ch_boot = pd.DataFrame()
            for con in np.unique(np.array([con for con in df_cha['Condition']])):
                print(con)
                print(df_cha[(df_cha['ch_name'] == ch['Unnamed: 0'][7:]) & (df_cha['Condition'] == con)])
                df_ch_boot = df_ch_boot.append(df_cha[(df_cha['ch_name'] == ch['Unnamed: 0'][7:]) & (df_cha['Condition'] == con)])
            sub_sorted_condition = dict_contrast_subcondition[contrast]

            fig, df_CI = helper_plot.plot_boxes_errorbar(sub_sorted_condition, df_ch_boot, 'Condition',  ['theta'], 'mean',
                                boot_size=5000, title=ch['Unnamed: 0'][7:].replace('_', '-').upper(), fwr_correction=True,
                                contrasts=False, figsize = (5, 4))

            save_path = os.path.join(save_directory, pick, contrast)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            df_CI.to_csv(os.path.join(save_path, 'Bootstrapping_' + ch['Unnamed: 0'][7:] +'.csv'), sep=';', decimal=',',
                         index=True, header=True)
            plt.show()
            fig.savefig(os.path.join(save_path, 'Bootstrapping_' + ch['Unnamed: 0'][7:] +'.svg'))


