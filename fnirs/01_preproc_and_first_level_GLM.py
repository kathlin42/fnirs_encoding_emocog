# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 18:42:07 2021

@author: hirning
"""
# =============================================================================
# Import required packages
# =============================================================================
import os
from collections import defaultdict
from copy import deepcopy

import mne

import numpy as np
import pandas as pd
os.chdir('../..')
from toolbox import config_analysis
from toolbox import helper_proc
# =============================================================================
# Paths and Variables
# =============================================================================
data_directory = os.path.join(config_analysis.project_directory, 'sourcedata')
analysis_settings = 'fNIRS_GLM_window_'
include_silence = '_correct_silence'#'_include_silence'
save_directory = os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_preproc', analysis_settings + str(config_analysis.GLM_time_window) + include_silence)
if not os.path.exists("{}".format(save_directory)):
    print('creating path for saving')
    os.makedirs("{}".format(save_directory))
#Exclude Subjects Pilot and First _VP01
subj_list = [f for f in os.listdir(os.path.join(data_directory, 'fnirs')) if
             not (f.startswith('.') or f.startswith('_') or f.startswith('Pilot'))]
subj_list.sort()

df_cha = pd.DataFrame()  # To store channel level results
df_con = pd.DataFrame()  # To store channel level contrast results
df_snr = pd.DataFrame()

total_trials = 0
dropped_epochs = 0
dropped_trials = 0

for subj, subj_folder in enumerate(subj_list):
    # subj = 11
    # subj_folder = subj_list[subj]
    subj += 2

    print(f"Subject: {subj}")
    fnirs_folders = [f for f in os.listdir(os.path.join(data_directory, 'fnirs', subj_folder)) if
                     not (f.startswith('.') or f.startswith('_'))]
    fnirs_folders.sort()
    raw_list = []
    event_list = []


    for i, fnirs_path in enumerate(fnirs_folders):
        # i = 0
        # fnirs_path = fnirs_folders[i]
        # Technical Issue -> Empty Data
        if (subj == 6) and (fnirs_path == '2021-06-04_003'):
            continue
        if (subj == 16) and (fnirs_path == '2021-06-10_002'):
            continue
        fnirs_file = os.path.join(data_directory, 'fnirs', subj_folder, fnirs_path, fnirs_path + '.snirf')
        raw = mne.io.read_raw_snirf(fnirs_file, verbose=True).load_data()
        events, num_trials = helper_proc.get_triggers(data_directory, subj_folder, fnirs_path, raw, subj, drop_silence = False)

        raw_list.append(raw)
        event_list.append(events)

    raw, events = mne.concatenate_raws(raws=raw_list, preload=True, events_list=event_list)
    events = events[events[:, 0] > 0]
    raw, event_dict = helper_proc.correct_annotations(raw, events, drop_silence = False)
    raw_haemo, channel, contrasts, dict_snr = helper_proc.first_level_GLM_analysis(raw, event_dict, events, subj, save_directory)
    if subj == 2:
        raw_haemo.save(os.path.join(save_directory, 'exemplary_raw.fif'),
            overwrite=True)

    total_trials += num_trials

    # Append individual results to dataframes
    df_cha = df_cha.append(channel)
    df_con = df_con.append(contrasts)
    df_snr = df_snr.append(pd.DataFrame(dict_snr))

df_snr['total_trials'] = total_trials

print(f"Mean CNR_HBO: {df_snr['CNR_HBO'].mean()}")
print(f"Mean CNR_HBR: {df_snr['CNR_HBR'].mean()}")
print(f"Mean SNR_PRE_OD: {df_snr['SNR_PRE_OD'].mean()}")
print(f"Mean SNR_PRE_RAW: {df_snr['SNR_PRE_RAW'].mean()}")
print(f"Mean SNR_POST_OD: {df_snr['SNR_POST_OD'].mean()}")

conditions = [key for key in list(event_dict.keys()) if key not in ['Baseline', 'Rest']]
df_cha = df_cha.loc[(df_cha['Condition'].isin(conditions))]
df_cha = df_cha[['ID', 'Condition', 'ch_name', 'Chroma', 'theta', 'se', 't', 'df', 'p_value', 'Source', 'Detector']]
df_cha.to_csv(os.path.join(save_directory, 'nirs_glm_cha.csv'),
              index=False, decimal=',', sep=';')

df_con = df_con[['ID', 'Contrast', 'ch_name', 'Chroma', 'effect', 'p_value', 'stat', 'z_score', 'Source', 'Detector']]
df_con.to_csv(os.path.join(save_directory, 'nirs_glm_con.csv'),
              index=False, decimal=',', sep=';')

df_snr.to_csv(os.path.join(save_directory, 'df_snr.csv'),
              index=False, decimal=',', sep=';')
