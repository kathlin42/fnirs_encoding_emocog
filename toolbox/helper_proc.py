import os
import re
import numpy as np
import pandas as pd

from itertools import compress
import mne
import mne_nirs
from nilearn.plotting import plot_design_matrix

import matplotlib.pyplot as plt
from toolbox import config_analysis

# =============================================================================
# Calculate Contrasts for Behavioral Data
# =============================================================================
def calc_contrasts_behav_data(df, subj_list, var_list):
    df_diff = pd.DataFrame(columns=['subj', 'contrast'] + var_list)
    for subj in subj_list:
        # Single Effects
        for cond_pair in config_analysis.single_effects:
            df_temp = pd.DataFrame(columns=['subj', 'contrast'] + var_list)
            df_subj = df.loc[(df['VP'] == subj) & (df['condition'].isin(cond_pair))]
            for variable in var_list:
                c1 = df_subj.loc[(df_subj['condition'] == cond_pair[0]), variable].values
                c1 = np.mean(c1[np.logical_not(np.isnan(c1))])
                c2 = df_subj.loc[(df_subj['condition'] == cond_pair[1]), variable].values
                c2 = np.mean(c2[np.logical_not(np.isnan(c2))])
                df_temp.loc[0, 'subj'] = subj
                df_temp.loc[0, 'contrast'] = str(cond_pair[0]) + '-' + str(cond_pair[1])
                df_temp.loc[df_temp['subj'] == subj, variable] = c1 - c2
            df_diff = pd.concat([df_diff, df_temp], ignore_index=True)

        # Interactions
        for cond_pair in config_analysis.interaction_effects:
            df_temp = pd.DataFrame(columns=['subj', 'contrast'] + var_list)
            df_subj = df_diff.loc[
                (df_diff['subj'] == subj) & (df_diff['contrast'].isin(cond_pair))]
            for variable in var_list:
                c1 = df_subj.loc[(df_subj['contrast'] == cond_pair[0]), variable].item()
                c2 = df_subj.loc[(df_subj['contrast'] == cond_pair[1]), variable].item()
                df_temp.loc[0, 'subj'] = subj
                df_temp.loc[0, 'contrast'] = str(cond_pair[0]) + '-' + str(cond_pair[1])
                df_temp.loc[df_temp['subj'] == subj, variable] = c1 - c2
            df_diff = pd.concat([df_diff, df_temp], ignore_index=True)

        # Main Effects
        for cond_pair in config_analysis.main_emotion_effects:
            df_temp = pd.DataFrame(columns=['subj', 'contrast'] + var_list)
            df_subj = df.loc[(df['VP'] == subj) & (df['Emotion_Condition'].isin(cond_pair))]
            for variable in var_list:
                c1 = df_subj.loc[(df_subj['Emotion_Condition'] == cond_pair[0]), variable].values
                c1 = np.mean(c1[np.logical_not(np.isnan(c1))])
                c2 = df_subj.loc[(df_subj['Emotion_Condition'] == cond_pair[1]), variable].values
                c2 = np.mean(c2[np.logical_not(np.isnan(c2))])
                df_temp.loc[0, 'subj'] = subj
                df_temp.loc[0, 'contrast'] = str(cond_pair[0][0:3]) + '-' + str(cond_pair[1][0:3])
                df_temp.loc[df_temp['subj'] == subj, variable] = c1 - c2
            df_diff = pd.concat([df_diff, df_temp], ignore_index=True)

        for cond_pair in config_analysis.main_workload_effects:
            df_temp = pd.DataFrame(columns=['subj', 'contrast'] + var_list)
            df_subj = df.loc[(df['VP'] == subj) & (df['Load_Condition'].isin(cond_pair))]
            for variable in var_list:
                c1 = df_subj.loc[(df_subj['Load_Condition'] == cond_pair[0]), variable].values
                c1 = np.mean(c1[np.logical_not(np.isnan(c1))])
                c2 = df_subj.loc[(df_subj['Load_Condition'] == cond_pair[1]), variable].values
                c2 = np.mean(c2[np.logical_not(np.isnan(c2))])
                df_temp.loc[0, 'subj'] = subj
                df_temp.loc[0, 'contrast'] = str(cond_pair[0]) + '-' + str(cond_pair[1])
                df_temp.loc[df_temp['subj'] == subj, variable] = c1 - c2
            df_diff = pd.concat([df_diff, df_temp], ignore_index=True)

    return df_diff

# =============================================================================
# Extract fNIRS trigger
# =============================================================================
def get_triggers(data_directory, subj_folder, fnirs_path, raw, subj):
    conditions = []
    samples = []
    df_mne = pd.DataFrame(columns=['Sample', '0', 'Condition'])
    with open(os.path.join(data_directory, 'fnirs', subj_folder, fnirs_path, fnirs_path + '_lsl.tri')) as f:
        content = f.readlines()
    for line in content:
        condition = int(re.findall(r';(\w+)', line)[-1])
        sample = int(re.findall(r';(\w+)', line)[0])
        conditions.append(condition)
        samples.append(sample)
    df_mne['Sample'] = samples
    df_mne['0'] = 0
    df_mne['Condition'] = conditions
    #Delete Conditions with Silence 35 = LowSil, 45 = HighSil
    df_mne = df_mne.loc[~df_mne['Condition'].isin([35, 45])]
    events = np.array(df_mne.sort_values(by='Sample')).astype(int)

    event_desc = {v: k for k, v in config_analysis.event_dict.items()}
    annotations = mne.annotations_from_events(events,
                                              raw.info['sfreq'],
                                              event_desc=event_desc)
    raw.set_annotations(annotations)

    df_events = pd.read_csv(os.path.join(data_directory, 'trigger', 'events_trial.csv'), sep=';', decimal=',')
    df_events = df_events.dropna(subset=['Timestamp_Aurora'])
    df_events = df_events.loc[df_events['Subject'] == subj].reset_index(drop=True)
    df_events = df_events[['Timestamp_Aurora', 'StimuliName']]
    # Delete Conditions with Silence 35 = LowSil, 45 = HighSil
    df_events = df_events.loc[~df_events['StimuliName'].isin(['LowSil', 'HighSil'])]
    num_trials = len(df_events)

    onset = raw.annotations.onset[0] * 1000
    df_events = df_events[(df_events['Timestamp_Aurora'] == onset).idxmax():].reset_index(drop=True)
    for i, row in df_events.iterrows():
        end = len(df_events) - 1
        if i + 2 == len(df_events):
            break
        if df_events.loc[i, ['Timestamp_Aurora']].item() > df_events.loc[i + 1, ['Timestamp_Aurora']].item():
            end = i
            break
    df_events = df_events[:end + 1]

    df_time_samp = pd.DataFrame({'Timestamp': raw.times * 1000,
                                 'Sample': np.arange(0, len(raw.times))})
    df_events = pd.merge_asof(df_events, df_time_samp, left_on='Timestamp_Aurora', right_on='Timestamp',
                              direction='nearest', tolerance=100)

    df_events['Condition'] = df_events['StimuliName'].replace(config_analysis.event_dict)

    df_mne = pd.DataFrame(columns=['Sample', '0', 'Condition'])
    df_mne['Sample'] = df_events['Sample']
    df_mne['0'] = 0
    df_mne['Condition'] = df_events['Condition']
    events = np.array(df_mne.sort_values(by='Sample')).astype(int)

    return events, num_trials


def correct_annotations(raw, events):
    event_desc = {v: k for k, v in config_analysis.event_dict.items()}
    # Only use Events in File
    event_desc = {k: event_desc[k] for k in pd.DataFrame(events)[2]}
    event_dict = {v: int(k) for k, v in event_desc.items()}

    annotations = mne.annotations_from_events(events,
                                              raw.info['sfreq'],
                                              event_desc=event_desc)
    annotations_new = annotations[annotations.onset > 0]
    raw.set_annotations(annotations_new)

    onsets = []
    durations = []
    descriptions = []
    for ann in raw.annotations:
        onsets.append(ann['onset'])
        descriptions.append(ann['description'])
        if ann['description'] == 'Rest':
            durations.append(config_analysis.time_rest)
        else:
            durations.append(config_analysis.GLM_time_window)

    new_annot = mne.Annotations(onset=onsets,  # in seconds
                                duration=durations,  # in seconds
                                description=descriptions)
    raw.set_annotations(new_annot)

    return raw, event_dict


def first_level_GLM_analysis(raw, event_dict, events, subj, save_directory):
    # raw.plot(duration=300, n_channels=len(raw_concat.ch_names))
    conditions = list(event_dict.keys())
    if event_dict != config_analysis.event_dict:
        print(f'ERROR - Not all conditions recorded for subject: {subj}')
        return

    conditions.remove('Rest')
    # Signal-to-Noise Ratio
    snr_pre_raw = (np.nanmean(np.ma.masked_invalid(raw.get_data())) / np.nanstd(np.ma.masked_invalid(raw.get_data())))
    print(f'Signal-to-Noise Ratio PRE RAW: {snr_pre_raw}')
    # Converting from raw intensity to optical density
    raw_od = mne.preprocessing.nirs.optical_density(raw)

    # Signal-to-Noise Ratio
    snr_pre_od = np.nanmean(np.ma.masked_invalid(raw_od.get_data())) / np.nanstd(np.ma.masked_invalid(raw_od.get_data()))
    print(f'Signal-to-Noise Ratio PRE OD: {snr_pre_od}')

    # Determine bad channels using scalp coupling index
    sci = mne.preprocessing.nirs.scalp_coupling_index(raw_od, l_freq=0.7, h_freq=1.5)
    bads = list(compress(raw_od.ch_names, sci < 0.5))
    raw_od.info['bads'] = bads

    # Repairs temporal derivative distribution
    raw_od = mne.preprocessing.nirs.tddr(raw_od)
    raw_od.info['bads'] = raw_od.info['bads'] + list(compress(raw_od.ch_names, np.isnan(raw_od.get_data()[:, 0])))

    # Signal-to-Noise Ratio on od post
    snr_post_od = np.nanmean(np.ma.masked_invalid(raw_od.get_data())) / np.nanstd(np.ma.masked_invalid(raw_od.get_data()))
    print(f'Signal-to-Noise Ratio POST OD: {snr_post_od}')

    # Converting from optical density to haemoglobin (Homer uses ppf=6, MNE pp=0.1)
    raw_haemo = mne.preprocessing.nirs.beer_lambert_law(raw_od, ppf=6)
    # raw_haemo.plot(n_channels=len(raw_haemo.ch_names), duration=300)
    raw_haemo.info['bads'] = raw_haemo.info['bads'] + list(compress(raw_haemo.ch_names, np.isnan(raw_haemo.get_data()[:, 0])))

    # Removing heart rate from signal using a low pass filter, removing slow drifts in the data using a high pass filter
    # fig = raw_haemo.plot_psd(average=True)
    # fig.suptitle('Before filtering', weight='bold', size='x-large')
    # fig.subplots_adjust(top=0.88)
    raw_haemo = raw_haemo.filter(method='iir', l_freq=0.05, h_freq=0.7, h_trans_bandwidth=0.2, l_trans_bandwidth=0.02)
    # fig = raw_haemo.plot_psd(average=True)
    # fig.suptitle('After filtering', weight='bold', size='x-large')
    # fig.subplots_adjust(top=0.88)
    # Negative correlation enhancement algorithm
    raw_haemo = mne_nirs.signal_enhancement.enhance_negative_correlation(raw_haemo)

    # Cut out just the short channels for creating a GLM repressor
    short_chans = mne_nirs.channels.get_short_channels(raw_haemo)
    raw_haemo = mne_nirs.channels.get_long_channels(raw_haemo)
    # raw_haemo.plot(n_channels=len(raw_haemo.ch_names), duration=300)

    # =============================================================================
    # Extract Epochs
    # =============================================================================
    # fig = mne.viz.plot_events(events, event_id=event_dict, sfreq=raw_haemo.info['sfreq'])
    # fig.subplots_adjust(right=0.7)  # make room for the legend
    reject_criteria = dict(hbo=80e-6)
    epochs = mne.Epochs(raw_haemo,
                        events,
                        event_id=event_dict,
                        tmin=0, tmax=config_analysis.GLM_time_window,
                        reject=reject_criteria,
                        reject_by_annotation=True,
                        # picks=mne.pick_types(raw_haemo.info, meg=False, fnirs=True, exclude=['bads']),
                        proj=True,
                        baseline=None,
                        detrend=0,
                        verbose=True,
                        preload=True)
    epochs.info['subject_info']['ID'] = subj
    # epochs.plot_drop_log()

    # Contrast-to-Noise-Ratio
    rest_mean_hbo = np.nanmean(epochs.copy().pick(picks="hbo")['Rest'].get_data())
    rest_mean_hbr = np.nanmean(epochs.copy().pick(picks="hbr")['Rest'].get_data())
    cond_mean_hbo = np.nanmean(epochs.copy().pick(picks="hbo")[conditions].get_data())
    cond_mean_hbr = np.nanmean(epochs.copy().pick(picks="hbr")[conditions].get_data())
    rest_std_hbo = np.nanstd(epochs.copy().pick(picks="hbo")['Rest'].get_data())
    rest_std_hbr = np.nanstd(epochs.copy().pick(picks="hbr")['Rest'].get_data())
    cond_std_hbo = np.nanstd(epochs.copy().pick(picks="hbo")[conditions].get_data())
    cond_std_hbr = np.nanstd(epochs.copy().pick(picks="hbr")[conditions].get_data())

    cnr_hbo = (cond_mean_hbo - rest_mean_hbo) / np.sqrt(cond_std_hbo + rest_std_hbo)
    cnr_hbr = (cond_mean_hbr - rest_mean_hbr) / np.sqrt(cond_std_hbr + rest_std_hbr)
    print(f'Contrast-to-Noise-Ratio - HbO: {cnr_hbo}')
    print(f'Contrast-to-Noise-Ratio - HbR: {cnr_hbr}')

    # =============================================================================
    # Plot Boxcar
    # =============================================================================
    # s = mne_nirs.experimental_design.create_boxcar(raw_haemo)
    # fig, axes = plt.subplots(figsize=(15, 6))
    # plt.plot(raw_haemo.times, s, axes=axes)
    # labels = np.unique(raw_haemo.annotations.description)
    # plt.legend(labels, loc="upper right")
    # plt.xlabel("Time (s)")
    # axes.axes.get_yaxis().set_ticks([])

    # =============================================================================
    # GLM
    # =============================================================================
    design_matrix = mne_nirs.experimental_design.make_first_level_design_matrix(raw_haemo,
                                                                                hrf_model='spm',  # 'glover'
                                                                                stim_dur=config_analysis.GLM_time_window,
                                                                                drift_order=3,
                                                                                drift_model='polynomial')

    # Append short channels mean to design matrix
    design_matrix["ShortHbO"] = np.nanmean(short_chans.copy().pick(picks="hbo").get_data(), axis=0)
    design_matrix["ShortHbR"] = np.nanmean(short_chans.copy().pick(picks="hbr").get_data(), axis=0)

    if subj == 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = plot_design_matrix(design_matrix, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), size=14, family='Times New Roman', weight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), size=14, family='Times New Roman', weight='bold')
        ax.set_ylabel(ax.get_ylabel().upper(), size=14, family='Times New Roman', weight='bold')
        fig.suptitle('Illustrative Example of a GLM Design Matrix', size=18, family='Times New Roman', weight='bold')

        fig.tight_layout()
        if not os.path.exists(os.path.join(save_directory, 'plots')):
            os.makedirs(os.path.join(save_directory, 'plots'))
        fig.savefig(os.path.join(save_directory, 'plots', 'Design_matrix_' + str(subj) + '.svg'))
        plt.close()

    # Run GLM
    glm_est = mne_nirs.statistics.run_glm(raw_haemo, design_matrix)

    # Extract channel metrics (per condition per channel per chroma, aggregated over epochs)
    cha = glm_est.to_dataframe()

    # Define contrasts for conditions
    defined_contrasts = []
    contrasts_names = []
    contrast_matrix = np.eye(design_matrix.shape[1])
    basic_conts = dict([(column, contrast_matrix[i]) for i, column in enumerate(design_matrix.columns)])

    basic_conts['main_wl_low'] = (basic_conts['LowNeu'] + basic_conts['LowPos'] + basic_conts['LowNeg'])
    basic_conts['main_wl_high'] = (basic_conts['HighNeu'] + basic_conts['HighPos'] + basic_conts['HighNeg'])
    basic_conts['main_emo_neu'] = (basic_conts['LowNeu'] + basic_conts['HighNeu'])
    basic_conts['main_emo_pos'] = (basic_conts['LowPos'] + basic_conts['HighPos'])
    basic_conts['main_emo_neg'] = (basic_conts['LowNeg'] + basic_conts['HighNeg'])

    # Main Contrasts
    main_wl_HighLow = basic_conts['main_wl_high'] - basic_conts['main_wl_low']
    defined_contrasts.append(main_wl_HighLow)
    contrasts_names.append('main_wl_HighLow')

    main_emo_NegPos = basic_conts['main_emo_neg'] - basic_conts['main_emo_pos']
    defined_contrasts.append(main_emo_NegPos)
    contrasts_names.append('main_emo_NegPos')
    main_emo_NeuPos =  basic_conts['main_emo_neu'] - basic_conts['main_emo_pos']
    defined_contrasts.append(main_emo_NeuPos)
    contrasts_names.append('main_emo_NeuPos')
    main_emo_NeuNeg = basic_conts['main_emo_neu'] - basic_conts['main_emo_neg']
    defined_contrasts.append(main_emo_NeuNeg)
    contrasts_names.append('main_emo_NeuNeg')

    # Contrasts
    contrast_low_NegPos = basic_conts['LowNeg'] - basic_conts['LowPos']
    defined_contrasts.append(contrast_low_NegPos)
    contrasts_names.append('contrast_low_NegPos')
    contrast_low_NeuPos = basic_conts['LowNeu'] - basic_conts['LowPos']
    defined_contrasts.append(contrast_low_NeuPos)
    contrasts_names.append('contrast_low_NeuPos')
    contrast_low_NeuNeg = basic_conts['LowNeu'] - basic_conts['LowNeg']
    defined_contrasts.append(contrast_low_NeuNeg)
    contrasts_names.append('contrast_low_NeuNeg')

    contrast_high_NegPos = basic_conts['HighNeg'] - basic_conts['HighPos']
    defined_contrasts.append(contrast_high_NegPos)
    contrasts_names.append('contrast_high_NegPos')
    contrast_high_NeuPos = basic_conts['HighNeu'] - basic_conts['HighPos']
    defined_contrasts.append(contrast_high_NeuPos)
    contrasts_names.append('contrast_high_NeuPos')
    contrast_high_NeuNeg = basic_conts['HighNeu'] - basic_conts['HighNeg']
    defined_contrasts.append(contrast_high_NeuNeg)
    contrasts_names.append('contrast_high_NeuNeg')

    contrast_pos_HighLow = basic_conts['HighPos'] - basic_conts['LowPos']
    defined_contrasts.append(contrast_pos_HighLow)
    contrasts_names.append('contrast_pos_HighLow')
    contrast_neu_HighLow = basic_conts['HighNeu'] - basic_conts['LowNeu']
    defined_contrasts.append(contrast_neu_HighLow)
    contrasts_names.append('contrast_neu_HighLow')
    contrast_neg_HighLow = basic_conts['HighNeg'] - basic_conts['LowNeg']
    defined_contrasts.append(contrast_neg_HighLow)
    contrasts_names.append('contrast_neg_HighLow')

    # Interactions
    contrast_inter_WL_NeuPos = contrast_neu_HighLow - contrast_pos_HighLow
    defined_contrasts.append(contrast_inter_WL_NeuPos)
    contrasts_names.append('contrast_inter_WL_NeuPos')
    contrast_inter_WL_NegPos = contrast_neg_HighLow - contrast_pos_HighLow
    defined_contrasts.append(contrast_inter_WL_NegPos)
    contrasts_names.append('contrast_inter_WL_NegPos')
    contrast_inter_WL_NeuNeg = contrast_neu_HighLow - contrast_neg_HighLow
    defined_contrasts.append(contrast_inter_WL_NeuNeg)
    contrasts_names.append('contrast_inter_WL_NeuNeg')

    contrast_inter_EMO_NeuPos = contrast_high_NeuPos - contrast_low_NeuPos
    defined_contrasts.append(contrast_inter_EMO_NeuPos)
    contrasts_names.append('contrast_inter_EMO_NeuPos')
    contrast_inter_EMO_NegPos = contrast_high_NegPos - contrast_low_NegPos
    defined_contrasts.append(contrast_inter_EMO_NegPos)
    contrasts_names.append('contrast_inter_EMO_NegPos')
    contrast_inter_EMO_NeuNeg = contrast_high_NeuNeg - contrast_low_NeuNeg
    defined_contrasts.append(contrast_inter_EMO_NeuNeg)
    contrasts_names.append('contrast_inter_EMO_NeuNeg')

    # Compute defined contrast
    con = pd.DataFrame()
    con['Contrast'] = None
    for c, name in zip(defined_contrasts, contrasts_names):
        contrast = glm_est.compute_contrast(c)
        df_temp = contrast.to_dataframe()
        df_temp['Contrast'] = name
        con = con.append(df_temp)

    # Add the participant ID to the dataframes
    cha["ID"] = con["ID"] = subj
    # Convert to uM for nicer plotting below.
    cha["theta"] = [t * 1.e6 for t in cha["theta"]]

    if len(con) > 0:
        con["effect"] = [t * 1.e6 for t in con["effect"]]

    dict_snr_cnr = {'ID': [subj],
                    'SNR_PRE_RAW': [snr_pre_raw],
                    'SNR_PRE_OD': [snr_pre_od],
                    'SNR_POST_OD': [snr_post_od],
                    'CNR_HBO': [cnr_hbo],
                    'CNR_HBR': [cnr_hbr],
                    'bads': [bads],
                    'n_bads': [len(bads)],
                    'n_epochs': [epochs._data.shape[0]]
                    }

    return raw_haemo, cha, con, dict_snr_cnr