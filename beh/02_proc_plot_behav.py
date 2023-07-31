# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 18:42:07 2021

@author: hirning
"""
# =============================================================================
# Import required packages
# =============================================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.chdir('..')
from toolbox import config_analysis
from toolbox import helper_proc
from toolbox import helper_plot

# =============================================================================
# Paths
# =============================================================================
data_directory = os.path.join(config_analysis.project_directory, 'sourcedata', 'performance')
save_path = os.path.join(config_analysis.project_directory, 'derivatives', 'beh_plots')
if not os.path.exists(save_path):
    os.makedirs(save_path)

fig_format = '.svg'
fig_format_pub = '.jpg'
# =============================================================================
# Read Data
# =============================================================================
df = pd.read_csv(os.path.join(data_directory, 'performance_trial.csv'), sep=';', decimal=',')
df["condition"] = df["condition"].astype("str")
df = df.loc[~(df["condition"].str.contains("Baseline"))]
df = df.loc[~(df["condition"].str.contains("Sil"))]
list_all_var = ['arousal', 'valence', 'effort', 'reaction time', 'accuracy']
# =============================================================================
# Calculate Subjective Contrasts: Blockwise (Valence & Arousal)
# =============================================================================
subj_list = [int(f[-2:]) for f in os.listdir(os.path.join(config_analysis.project_directory, 'sourcedata', 'fnirs')) if not (f.startswith('.') or f.startswith('_'))]
subj_list.sort()
df_block = df.drop(columns=['trial', 'Aggregated Performance Score', 'accuracy', 'speed'])
df_block = df_block.groupby(["VP", "block", "condition", "Load_Condition", "Emotion_Condition"]).mean().reset_index()
df_block = df_block.dropna(subset=['valence', 'arousal', 'effort'])
df_block['Emotion_Condition'] = df_block['Emotion_Condition'].replace({'Negative':'Neg', 'Neutral':'Neu', 'Positive': 'Pos'})
df_diff_subjective = helper_proc.calc_contrasts_behav_data(df_block, subj_list, ['valence', 'arousal', 'effort'])
# =============================================================================
# Calculate Subjective Contrasts: Trialwise (Accuracy & RT)
# =============================================================================
df_trial = df.drop(columns=['Aggregated Performance Score', 'effort', 'frustration', 'valence', 'arousal', 'CAAT - Emotion'])
df_trial = df_trial.dropna(subset=['speed', 'accuracy'])
df_trial['speed'] = [(x - np.min(df_trial['speed'])) / (np.max(df_trial['speed']) - np.min(df_trial['speed'])) for x in df_trial['speed']]
df_trial['accuracy'] = [(x - np.min(df_trial['accuracy'])) / (np.max(df_trial['accuracy']) - np.min(df_trial['accuracy'])) for x in df_trial['accuracy']]

df_trial['Emotion_Condition'] = df_trial['Emotion_Condition'].replace({'Negative':'Neg', 'Neutral':'Neu', 'Positive': 'Pos'})
df_diff_perf = helper_proc.calc_contrasts_behav_data(df_trial, subj_list,['speed', 'accuracy'])
# =============================================================================
# Merge and Clean Subjective and Performance
# =============================================================================
df_plot = df_diff_subjective.merge(df_diff_perf, on=["subj", "contrast"])
df_plot[['valence', 'arousal', 'effort', 'speed', 'accuracy']] = df_plot[['valence', 'arousal', 'effort', 'speed', 'accuracy']].astype("float")
df_plot = df_plot.rename(columns={"speed": "reaction time"})
# =============================================================================
# Single Effects
# =============================================================================
fig_title = 'Subjective Ratings and Performance - Contrasts of Condition Effects'
df_full, fig = helper_plot.plot_colored_errorbar([cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.single_effects],
                                                 df_plot,
                                                 col_group="contrast",
                                                 labels=list_all_var,
                                                 boot='mean',
                                                 boot_size=5000,
                                                 title=fig_title,
                                                 lst_color=['#09D21D','#09D5AD', '#15A3D7'],
                                                 fwr_correction=True,
                                                 contrasts=True,
                                                 n_col=3,
                                                 figsize=(18, 10), fs=20,
                                                 reduced_text=True,
                                                 grouped_colors=True,
                                                 groupsize=3)
fig.savefig(os.path.join(save_path, 'Supplementary_Figure_5_Bootstrapped_perf_subj_single_effects' + fig_format_pub), dpi=700)
plt.close()
df_full = df_full.reset_index()
df_full['Variable'] = list(np.repeat(list_all_var, len([cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.single_effects])))
df_full.to_csv(os.path.join(save_path, 'Supplementary_Figure_5_Bootstrapped_perf_subj_single_effects.csv'), header=True, index=False, sep =';', decimal=',')
# =============================================================================
# Main Effects for Subjective Emo Ratings
# =============================================================================
list_plot_var = ['arousal', 'valence']
fig_title = 'Subjective Valence and Arousal - Emotion and Workload Effects'
df_full, fig = helper_plot.plot_colored_errorbar([cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.main_emotion_effects] +
                                                 [cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.main_workload_effects],
                                     df_plot,
                                     col_group="contrast",
                                     labels=list_plot_var,
                                     boot='mean',
                                     boot_size=5000,
                                     title=fig_title,
                                     lst_color=['#09D21D', '#09D5AD', '#15A3D7'],
                                     fwr_correction=True,
                                     contrasts=True,
                                     n_col=2,
                                     figsize=(10, 5), fs=18,
                                     reduced_text=False,
                                     grouped_colors=True,
                                     groupsize=3)
fig.savefig(os.path.join(save_path, 'Bootstrapped_val_arousal_main_effects' + fig_format))
plt.close()
df_full = df_full.reset_index()
df_full['Variable'] = list(np.repeat(list_plot_var, len([cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.main_emotion_effects] +
                                                        [cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.main_workload_effects])))
df_full.to_csv(os.path.join(save_path, 'Bootstrapped_val_arousal_main_effects.csv'), header=True, index=False, sep =';', decimal=',')
# =============================================================================
# Main Effects for Effort and Performance
# =============================================================================
list_plot_var = ['effort', 'reaction time', 'accuracy']
fig_title = 'Subjective Effort and Performance - Emotion and Workload Effects'
df_full, fig = helper_plot.plot_colored_errorbar([cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.main_emotion_effects] +
                                                 [cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.main_workload_effects],
                                     df_plot,
                                     col_group="contrast",
                                     labels=list_plot_var,
                                     boot='mean',
                                     boot_size=5000,
                                     title=fig_title,
                                     lst_color=['#09D21D', '#09D5AD', '#15A3D7'],
                                     fwr_correction=True,
                                     contrasts=True,
                                     n_col=3,
                                     figsize=(15, 6), fs=18,
                                     reduced_text=False,
                                     grouped_colors=True,
                                     groupsize=3)
fig.savefig(os.path.join(save_path, 'Bootstrapped_eff_perf_main_effects' + fig_format))
plt.close()
df_full = df_full.reset_index()
df_full['Variable'] = list(np.repeat(list_plot_var, len([cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.main_emotion_effects] +
                                                        [cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.main_workload_effects])))
df_full.to_csv(os.path.join(save_path, 'Bootstrapped_eff_perf_main_effects.csv'), header=True, index=False, sep =';', decimal=',')
# =============================================================================
# Main Effects for All Variables
# =============================================================================
fig_title = 'Subjective Ratings and Performance - Contrasts of Main Effects'
df_full, fig = helper_plot.plot_colored_errorbar([cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.main_emotion_effects] +
                                                 [cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.main_workload_effects],
                                     df_plot,
                                     col_group="contrast",
                                     labels=list_all_var,
                                     boot='mean',
                                     boot_size=5000,
                                     title=fig_title,
                                     lst_color=['#09D21D', '#09D5AD', '#15A3D7'],
                                     fwr_correction=True,
                                     contrasts=True,
                                     n_col=3,
                                     figsize=(12, 8), fs=18,
                                     reduced_text=False,
                                     grouped_colors=True,
                                     groupsize=3)
fig.savefig(os.path.join(save_path, 'Figure_6_Bootstrapped_perf_subj_main_effects' + fig_format))
fig.savefig(os.path.join(save_path, 'Figure_6_Bootstrapped_perf_subj_main_effects' + fig_format_pub))
plt.close()
df_full = df_full.reset_index()
df_full['Variable'] = list(np.repeat(list_all_var, len([cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.main_emotion_effects] +
                                                        [cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.main_workload_effects])))
df_full.to_csv(os.path.join(save_path, 'Figure_6_Bootstrapped_perf_subj_main_effects.csv'), header=True, index=False, sep =';', decimal=',')

# =============================================================================
# Interaction Effects
# =============================================================================
fig_title = 'Subjective Ratings and Performance - Contrasts of Interaction Effects'
df_full, fig = helper_plot.plot_colored_errorbar([cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.interaction_effects],
                                                 df_plot,
                                                 col_group="contrast",
                                                 labels=list_all_var,
                                                 boot='mean',
                                                 boot_size=5000,
                                                 title=fig_title,
                                                 lst_color=['#09D21D','#09D5AD', '#15A3D7'],
                                                 fwr_correction=True,
                                                 contrasts=True,
                                                 n_col=3,
                                                 figsize=(18, 10), fs=20,
                                                 reduced_text=True,
                                                 grouped_colors=False)
fig.savefig(os.path.join(save_path, 'Supplementary_Figure_4_Bootstrapped_perf_subj_interaction_effects' + fig_format_pub), dpi=700)
plt.close()
df_full = df_full.reset_index()
df_full['Variable'] = list(np.repeat(list_all_var, len([cond_pair[0] + '-' + cond_pair[1] for cond_pair in config_analysis.interaction_effects])))
df_full.to_csv(os.path.join(save_path, 'Supplementary_Figure_4_Bootstrapped_perf_subj_interaction_effects.csv'), header=True, index=False, sep =';', decimal=',')
df_plot.to_csv(os.path.join(save_path, 'perf_subj_results.csv'), header=True, index=False, sep =';', decimal=',')
