# =============================================================================
# Directories and Imports
# =============================================================================
import os
import re
import numpy as np
import pandas as pd
import mne
from toolbox import config_analysis

subj_list = [int(f[-2:]) for f in os.listdir(os.path.join(config_analysis.project_directory, 'sourcedata', 'fnirs')) if not (f.startswith('.') or f.startswith('_'))]
subj_list.sort()

df_events_all = pd.DataFrame(columns  = ['Subject', 'fileName', 'Load_Condition', 'Emotion_Condition', 'Event', 'Timestamp_Rate', 'Duration_ms'])
df_events_all_fnirs = pd.DataFrame(columns  = ['Subject', 'Block', 'Timestamp_Aurora', 'StimuliName'])
for subj, sub_folder in enumerate(subj_list):
    # subj = 0
    # sub_folder = subj_list[subj]
    subj += 2
    print(f'Subject: {subj}, {sub_folder}')
    df = pd.read_csv(os.path.join(config_analysis.project_directory, 'sourcedata', 'Rate', sub_folder + '.csv'), skiprows=33, header=0)
    df.loc[df['fileName'].str.contains("1", na=False), 'Trial'] = 1
    df.loc[df['fileName'].str.contains("2", na=False), 'Trial'] = 2
    df.loc[df['fileName'].str.contains("3", na=False), 'Trial'] = 3
    df = df.sort_values('Timestamp')

    # Check Duration
    df_events = df[['Timestamp', 'Trial', 'fileName']]
    df_events = df_events.loc[df_events['fileName'].notna()].reset_index(drop=True)
    # Drop consecutive duplicates of fileName, keep first
    df_events = df_events.loc[df_events['fileName'] != df_events['fileName'].shift()].sort_index().reset_index(drop=True)

    # Compute Duration per Event
    df_events['Timedelta'] = pd.to_timedelta(df_events['Timestamp'], 'ms')
    df_events['Duration'] = ''
    for i, row in df_events.iterrows():
        if i == len(df_events) - 1:
            continue
        df_events.loc[i, 'Duration'] = df_events.loc[i + 1, 'Timedelta'] - df_events.loc[i, 'Timedelta']
    df_events['Duration'] = pd.to_timedelta(df_events['Duration'])
    # Add ID
    df_events['VP'] = subj
    # Drop second fileName row
    df_events = df_events.loc[df_events['fileName'] != df_events['fileName'].shift()]
    df_events = df_events[['VP', 'fileName', 'Duration', 'Timestamp']].reset_index(drop=True)
    df_events['Duration_ms'] = df_events['Duration'].dt.total_seconds() * 1000
    df_events['Duration_ms'] = df_events['Duration_ms'].replace(np.nan, 0)
    df_events['Duration_ms'] = df_events['Duration_ms'].astype('int')

    df_events.rename(columns={'VP': 'Subject', 'Timestamp': 'Timestamp_Rate'}, inplace=True)
    df_events['Event'] = 'StartTrial'

    df_events.loc[df_events['fileName'].str.contains("High", na=False), 'Load_Condition'] = 'High'
    df_events.loc[df_events['fileName'].str.contains("Low", na=False), 'Load_Condition'] = 'Low'
    df_events.loc[df_events['fileName'].str.contains("Pos", na=False), 'Emotion_Condition'] = 'Positive'
    df_events.loc[df_events['fileName'].str.contains("Neg", na=False), 'Emotion_Condition'] = 'Negative'
    df_events.loc[df_events['fileName'].str.contains("Neu", na=False), 'Emotion_Condition'] = 'Neutral'
    df_events.loc[df_events['fileName'].str.contains("Sil", na=False), 'Emotion_Condition'] = 'Silence'
    df_events.loc[df_events['fileName'].str.contains("Baseline", na=False), 'Load_Condition'] = 'Baseline'
    df_events.loc[df_events['fileName'].str.contains("Baseline", na=False), 'Emotion_Condition'] = 'Silence'

    df_events = df_events[['Subject', 'fileName', 'Load_Condition', 'Emotion_Condition', 'Event', 'Timestamp_Rate', 'Duration_ms']]
    df_events_all = df_events_all.append(df_events)

    files = [f for f in os.listdir(os.path.join(config_analysis.project_directory, 'sourcedata', 'fnirs', sub_folder)) if not (f.startswith('.') or f.startswith('_'))]
    files.sort()

    block = 0
    for file in files:

        file_data = os.path.join(config_analysis.project_directory, 'sourcedata', 'fnirs', sub_folder, file, file + '.snirf')
        raw = mne.io.read_raw_snirf(file_data, verbose=True).load_data()

        conditions = []
        samples = []
        df_mne = pd.DataFrame(columns=['Sample', '0', 'Condition'])
        with open(os.path.join(config_analysis.project_directory, 'sourcedata', 'fnirs', sub_folder, file, file + '_lsl.tri')) as f:
            content = f.readlines()
        for line in content:
            condition = int(re.findall(r';(\w+)', line)[-1])
            sample = int(re.findall(r';(\w+)', line)[0])
            conditions.append(condition)
            samples.append(sample)
        df_mne['Sample'] = samples
        df_mne['0'] = 0
        df_mne['Condition'] = conditions

        events = np.array(df_mne.sort_values(by='Sample')).astype(int)
        event_desc = {v: k for k, v in config_analysis.event_dict.items()}
        # time_of_first_sample = raw.first_samp / raw.info['sfreq']
        annotations = mne.annotations_from_events(events,
                                                  raw.info['sfreq'],
                                                  event_desc=event_desc)
        raw.set_annotations(annotations)

        df_events_fnirs = pd.DataFrame({'Timestamp_Aurora': raw.annotations.onset * 1000,
                                        'StimuliName': raw.annotations.description})
        df_events_fnirs = df_events_fnirs.assign(Subject=subj)
        df_events_fnirs = df_events_fnirs.assign(Block=np.arange(block, block + len(df_events_fnirs)))
        block += len(df_events_fnirs)

        df_events_fnirs = df_events_fnirs[['Subject', 'Block', 'Timestamp_Aurora', 'StimuliName']]
        df_events_all_fnirs = df_events_all_fnirs.append(df_events_fnirs)

df_events_all.to_csv(os.path.join(config_analysis.project_directory, 'sourcedata', 'trigger', 'events_rate.csv'), index=False, header=True, decimal=',', sep=';')
df_events_all_fnirs.to_csv(os.path.join(config_analysis.project_directory, 'sourcedata', 'trigger', 'events_aurora.csv'), index=False, header=True, decimal=',', sep=';')

# =============================================================================
# Get Number of Trials
# =============================================================================
df_events_cleaned = pd.read_csv(os.path.join(config_analysis.project_directory, 'sourcedata', 'trigger', 'events_trial.csv'), sep=';', decimal=',')
trials_cleaned = len(df_events_cleaned.groupby(['Subject', 'Trial']).count())
trials_cleaned = len(df_events_cleaned.groupby(['Subject', 'Load_Condition']).count())
