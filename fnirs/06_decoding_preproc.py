# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 18:42:07 2021

@author: hirning
"""
# =============================================================================
# Import required packages
# =============================================================================
import os
import mne
import pandas as pd
import numpy as np
from collections import defaultdict
from copy import deepcopy
os.chdir('../..')
from toolbox import config_analysis
from toolbox import helper_proc
# =============================================================================
# Paths and Variables
# =============================================================================
data_directory = os.path.join(config_analysis.project_directory, 'sourcedata')
epoch_length = 10
analysis_settings = 'fNIRS_decoding_epoch_length_{}'.format(epoch_length)
include_silence = '_include_silence'#'_include_silence'
save_directory = os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_preproc', analysis_settings + str(config_analysis.GLM_time_window) + include_silence)
os.makedirs("{}".format(save_directory), exist_ok=True)

#Exclude Subjects Pilot and First _VP01
subj_list = [f for f in os.listdir(os.path.join(data_directory, 'fnirs')) if
             not (f.startswith('.') or f.startswith('_') or f.startswith('Pilot'))]
subj_list.sort()
df_meta = pd.DataFrame()
df_epochs = pd.DataFrame()

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
    events = helper_proc.resample_non_overlapping_mne_markers(events, raw.info['sfreq'], 60, 10)
    raw, event_dict, event_desc = helper_proc.correct_annotations(raw, events, drop_silence = False)

    epochs, dict_meta = helper_proc.preproc_and_extract_epochs(raw, event_dict, events, subj, save_directory)
    # Save individual-evoked participant data along with others in all_evokeds
    df_epochs_sub = epochs.to_data_frame()
    df_epochs_sub['ID'] = subj
    df_epochs = pd.concat((df_epochs, df_epochs_sub))
    dict_count = {'total_epochs': len(epochs), 'dropped_epochs_in_%': epochs.drop_log_stats()}
    for event in range(0, len(np.unique(events[:, 2]))):
        if event_desc[np.unique(events[:, 2], return_counts=True)[0][event]] in list(event_dict.keys()):
            dict_meta['n_trial_' + event_desc[np.unique(events[:, 2], return_counts=True)[0][event]]] = [\
                np.unique(events[:, 2], return_counts=True)[1][event]]
            if event_desc[np.unique(events[:, 2], return_counts=True)[0][event]] not in list(dict_count.keys()):
                dict_count['n_trial_' + event_desc[np.unique(events[:, 2], return_counts=True)[0][event]]] = \
                np.unique(events[:, 2], return_counts=True)[1][event]
            else:
                dict_count['n_trial_' + event_desc[np.unique(events[:, 2], return_counts=True)[0][event]]] += \
                np.unique(events[:, 2], return_counts=True)[1][event]
    # Append individual results to dataframes
    dict_meta.update(dict_count)
    df_meta = pd.concat((df_meta, pd.DataFrame(dict_meta)))
df_epochs.to_csv(os.path.join(save_directory,'df_epochs.csv'), index=False, header = True, decimal=',', sep=';')
