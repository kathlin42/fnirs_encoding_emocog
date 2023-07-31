# =============================================================================
# Directories and Imports
# =============================================================================
import os
import pandas as pd
from toolbox import config_analysis

subjects = [f for f in os.listdir(os.path.join(config_analysis.project_directory, 'sourcedata', 'fnirs')) if not (f.startswith('.') or f.startswith('_'))]
ids = [int(f[-2:]) for f in subjects]
ids.sort()

df_subjective = pd.read_csv(os.path.join(config_analysis.project_directory, 'sourcedata', 'questionnaires', 'MIKADO2_Break.csv'), sep=';', decimal=',', usecols=[0, 1, 2, 3, 4, 5, 6])
df_subjective = df_subjective[df_subjective['Condition'].str.contains('Training') == False]
df_subjective.loc[df_subjective['Condition'].str.contains("Baseline", na=False), 'Condition'] = 'Baseline'
df_subjective.loc[df_subjective['Condition'].str.contains("HighNeu", na=False), 'Condition'] = 'HighNeu'
df_subjective.loc[df_subjective['Condition'].str.contains("HighNeg", na=False), 'Condition'] = 'HighNeg'
df_subjective.loc[df_subjective['Condition'].str.contains("HighPos", na=False), 'Condition'] = 'HighPos'
df_subjective.loc[df_subjective['Condition'].str.contains("HighSil", na=False), 'Condition'] = 'HighSil'
df_subjective.loc[df_subjective['Condition'].str.contains("LowNeu", na=False), 'Condition'] = 'LowNeu'
df_subjective.loc[df_subjective['Condition'].str.contains("LowNeg", na=False), 'Condition'] = 'LowNeg'
df_subjective.loc[df_subjective['Condition'].str.contains("LowPos", na=False), 'Condition'] = 'LowPos'
df_subjective.loc[df_subjective['Condition'].str.contains("LowSil", na=False), 'Condition'] = 'LowSil'

df_complete_trial_all = pd.DataFrame(columns =['Subject', 'Block', 'Trial', 'Condition', 'Load_Condition', 'Emotion_Condition',
                       'Aggregated Performance Score', 'Accuracy', 'Speed', 'Nasa TLX - Anstrengung',
                       'Nasa TLX - Frustration', 'EmojiGrid - Valenz', 'EmojiGrid - Arousal', 'CAAT - Emotion'])
for (id, subject) in zip(ids, subjects):
    # subject = subjects[0]
    # id = ids[0]
    print(f'Subject: {id}, {subject}')
    df = pd.read_csv(os.path.join(config_analysis.project_directory, 'sourcedata', 'Rate', subject + '.csv'), skiprows=33, header=0)

    df['VP'] = subject
    df['Round'] = df['SourceStimuliName'].str[5]
    df.loc[df['fileName'].str.contains("High", na=False), 'Load_Condition'] = 'High'
    df.loc[df['fileName'].str.contains("Low", na=False), 'Load_Condition'] = 'Low'
    df.loc[df['fileName'].str.contains("Pos", na=False), 'Emotion_Condition'] = 'Positive'
    df.loc[df['fileName'].str.contains("Neg", na=False), 'Emotion_Condition'] = 'Negative'
    df.loc[df['fileName'].str.contains("Neu", na=False), 'Emotion_Condition'] = 'Neutral'
    df.loc[df['fileName'].str.contains("Sil", na=False), 'Emotion_Condition'] = 'Silence'
    df.loc[df['fileName'].str.contains("Baseline", na=False), 'Load_Condition'] = 'Baseline'
    df.loc[df['fileName'].str.contains("Baseline", na=False), 'Emotion_Condition'] = 'Silence'
    df.loc[df['fileName'].str.contains("HighNeu", na=False), 'StimuliName'] = 'HighNeu'
    df.loc[df['fileName'].str.contains("HighNeg", na=False), 'StimuliName'] = 'HighNeg'
    df.loc[df['fileName'].str.contains("HighPos", na=False), 'StimuliName'] = 'HighPos'
    df.loc[df['fileName'].str.contains("HighSil", na=False), 'StimuliName'] = 'HighSil'
    df.loc[df['fileName'].str.contains("LowNeu", na=False), 'StimuliName'] = 'LowNeu'
    df.loc[df['fileName'].str.contains("LowNeg", na=False), 'StimuliName'] = 'LowNeg'
    df.loc[df['fileName'].str.contains("LowPos", na=False), 'StimuliName'] = 'LowPos'
    df.loc[df['fileName'].str.contains("LowSil", na=False), 'StimuliName'] = 'LowSil'
    df.loc[df['fileName'].str.contains("Baseline", na=False), 'StimuliName'] = 'Baseline'

    # Rename Columns
    dict_names = {'EventSource': 'Scenario',
                  'EventSource.1': 'Face Recognition: Affectiva AFFDEX',
                  'EventSource.2': 'Face Recognition: Affectiva Landmarks',
                  'EventSource.3': 'Emotient FACET',
                  'EventSource.4': 'Belt Battery',
                  'EventSource.5': 'Belt',
                  'EventSource.6': 'Filename',
                  'EventSource.7': 'Score bekämpfen',
                  'EventSource.8': 'Score Experiment',
                  'EventSource.9': 'Score fachlich: Accuracy',
                  'EventSource.10': 'Score identifizieren',
                  'EventSource.11': 'Score Sequenz: Overall Performance',
                  'EventSource.12': 'Score warnen',
                  'EventSource.13': 'Score zeitlich: Speed',
                  'EventSource.21': 'Eye Tracking'}
    df.rename(columns=dict_names, inplace=True)

    # Drop Columns with irrelevant data
    df = df.drop(columns=df.columns[230:286])
    df = df.drop(columns=df.columns[218:224])
    df = df.drop(columns=df.columns[206:212])
    df = df.drop(columns=df.columns[188:200])
    df = df.drop(columns=df.columns[2:187])

    # Reorder Columns
    cols = df.columns
    cols = cols[-5:].tolist() + cols[:-5].tolist()
    df = df[cols]

    df = df.sort_values('Timestamp')

    # CONDITION
    df_condition = df[['VP', 'Round', 'Timestamp', 'fileName', 'StimuliName', 'Load_Condition', 'Emotion_Condition']]
    # Drop NaNs and Consecutive Duplicates of Condition Indicator
    df_condition = df_condition.loc[df_condition['fileName'].notna() & df_condition['StimuliName'].notna()]
    df_condition = df_condition.loc[df_condition['fileName'] != df_condition['fileName'].shift()].sort_index().reset_index(drop=True)
    df_condition = df_condition[['VP', 'Timestamp', 'fileName', 'StimuliName', 'Load_Condition', 'Emotion_Condition']]

    # Add Block index
    df_block = df_condition.loc[df_condition['StimuliName'] != df_condition['StimuliName'].shift()].sort_index().reset_index(drop=True)
    df_block['Block'] = df_block.index
    df_block = df_block[['Block', 'Timestamp']]

    df_events = pd.read_csv(os.path.join(config_analysis.project_directory, 'sourcedata', 'trigger', 'events_trial.csv'), sep=';', decimal=',')
    df_events = df_events.dropna(subset=['Timestamp_RATE'])
    df_events['Timestamp_RATE'] = df_events['Timestamp_RATE'].astype('int64')
    df_events = df_events.loc[df_events['Subject'] == id].reset_index(drop=True)

    df_merge = df.drop(columns=df.columns[0:6])
    df_merge = df_merge.drop(columns=['fileName'])

    df_merge = pd.merge_asof(df_merge, df_events[['Timestamp_RATE', 'Trial']].reset_index(drop=True),
                             left_on='Timestamp', right_on='Timestamp_RATE',
                             direction='backward')  # , tolerance=int(fs / 2))

    df_merge = pd.merge_asof(df_merge, df_events[['Timestamp_RATE', 'Event']].reset_index(drop=True),
                             left_on='Timestamp', right_on='Timestamp_RATE', direction='backward',
                             tolerance=1000)

    df_merge = pd.merge_asof(df_merge, df_block, on='Timestamp', direction='backward')

    df_merge = pd.merge_asof(df_merge, df_condition, on='Timestamp', direction='backward')

    # OVERALL PERFORMANCE
    # Drop NaNs and Consecutive Duplicates of Performance Indicator
    df_performance = df_merge.loc[df_merge['Score Sequenz: Overall Performance'].notna()]
    df_performance = df_performance[['VP', 'Timestamp', 'Block', 'Trial',
                                     'fileName', 'StimuliName', 'Load_Condition', 'Emotion_Condition',
                                     'positive.4', 'negative.4', 'summary.4', 'max.4', 'performance.4']]
    df_performance = df_performance.loc[df_performance['performance.4'] != df_performance['performance.4'].shift()]
    # Reset Index for Selection of relevant Rows
    df_performance = df_performance.reset_index(drop=True)
    # Recalculate performance indicator
    df_performance['performance_new'] = df_performance['summary.4'] / df_performance['max.4']
    df_performance = df_performance.loc[df_performance['performance_new'].notna()]
    # Drop consecutive duplicates of Trial, keep last
    df_performance_trial = df_performance.loc[
        df_performance['Trial'] != df_performance['Trial'].shift(-1)].sort_index().reset_index(drop=True)
    df_performance_trial = df_performance_trial[
        ['VP', 'Timestamp', 'Block', 'Trial', 'fileName', 'StimuliName', 'Load_Condition', 'Emotion_Condition',
         'performance_new']]

    # # Drop consecutive duplicates of Block, keep last
    # df_performance_block = df_performance.loc[
    #     df_performance['Block'] != df_performance['Block'].shift(-1)].sort_index().reset_index(drop=True)
    # df_performance_block = df_performance_block[
    #     ['VP', 'Timestamp', 'Block', 'Trial', 'fileName', 'StimuliName', 'Load_Condition', 'Emotion_Condition',
    #      'performance_new']]

    # PERFORMANCE: ACCURACY
    # Drop NaNs and Consecutive Duplicates of Performance Indicator
    df_acc = df_merge.loc[df_merge['Score fachlich: Accuracy'].notna()]
    df_acc = df_acc[['VP', 'Timestamp', 'Block', 'Trial',
                     'fileName', 'StimuliName', 'Load_Condition', 'Emotion_Condition',
                     'positive.2', 'negative.2', 'summary.2', 'max.2', 'performance.2']]
    df_acc = df_acc.loc[df_acc['performance.2'] != df_acc['performance.2'].shift()]
    # Reset Index for Selection of relevant Rows
    df_acc = df_acc.reset_index(drop=True)
    # Recalculate performance indicator
    df_acc['performance_acc'] = df_acc['summary.2'] / df_acc['max.2']
    df_acc = df_acc.loc[df_acc['performance_acc'].notna()]
    # Drop consecutive duplicates of Trial, keep last
    df_acc_trial = df_acc.loc[
        df_acc['Trial'] != df_acc['Trial'].shift(-1)].sort_index().reset_index(drop=True)
    df_acc_trial = df_acc_trial[['VP', 'Timestamp', 'Block', 'Trial', 'fileName', 'StimuliName', 'Load_Condition', 'Emotion_Condition',
         'performance_acc']]

    # # Drop consecutive duplicates of Block, keep last
    # df_acc_block = df_acc.loc[
    #     df_acc['Block'] != df_acc['Block'].shift(-1)].sort_index().reset_index(drop=True)
    # df_acc_block = df_acc_block[
    #     ['VP', 'Timestamp', 'Block', 'Trial', 'fileName', 'StimuliName', 'Load_Condition', 'Emotion_Condition',
    #      'performance_acc']]

    # PERFORMANCE: SPEED
    # Drop NaNs and Consecutive Duplicates of Performance Indicator
    df_speed = df_merge.loc[df_merge['Score zeitlich: Speed'].notna()]
    df_speed = df_speed[['VP', 'Timestamp', 'Block', 'Trial',
                         'fileName', 'StimuliName', 'Load_Condition', 'Emotion_Condition',
                         'positive.6', 'negative.6', 'summary.6', 'max.6', 'performance.6']]
    df_speed = df_speed.loc[df_speed['performance.6'] != df_speed['performance.6'].shift()]
    # Reset Index for Selection of relevant Rows
    df_speed = df_speed.reset_index(drop=True)
    # Recalculate performance indicator
    df_speed['performance_speed'] = df_speed['summary.6'] / df_speed['max.6']
    df_speed = df_speed.loc[df_speed['performance_speed'].notna()]
    # Drop consecutive duplicates of Trial, keep last
    df_speed_trial = df_speed.loc[
        df_speed['Trial'] != df_speed['Trial'].shift(-1)].sort_index().reset_index(drop=True)
    df_speed_trial = df_speed_trial[
        ['VP', 'Timestamp', 'Block', 'Trial', 'fileName', 'StimuliName', 'Load_Condition', 'Emotion_Condition',
         'performance_speed']]

    # # Drop consecutive duplicates of Block, keep last
    # df_speed_block = df_speed.loc[
    #     df_speed['Block'] != df_speed['Block'].shift(-1)].sort_index().reset_index(drop=True)
    # df_speed_block = df_speed_block[
    #     ['VP', 'Timestamp', 'Block', 'Trial', 'fileName', 'StimuliName', 'Load_Condition', 'Emotion_Condition',
    #      'performance_speed']]
    # Merge
    df_complete_trial = pd.merge_asof(df_performance_trial, df_acc_trial[['Timestamp', 'performance_acc']],
                                      on='Timestamp', direction='nearest')
    df_complete_trial = pd.merge_asof(df_complete_trial, df_speed_trial[['Timestamp', 'performance_speed']],
                                      on='Timestamp', direction='nearest')

    df_complete_trial.rename(columns={'StimuliName': 'Condition',
                                      'performance_new': 'Aggregated Performance Score',
                                      'performance_acc': 'Accuracy',
                                      'performance_speed': 'Speed'}, inplace=True)
    df_complete_trial = df_complete_trial.drop_duplicates(subset='Timestamp')

    # Merge with subjective evaluations
    df_complete_trial = pd.merge_asof(df_complete_trial.reset_index(drop=True),
                                      df_subjective.loc[df_subjective['Subject ID'] == id].reset_index(drop=True),
                                      left_index=True, right_index=True, by='Condition', direction='backward')

    # Rename and reorder columns, add correct subject ID
    column_names = ['Subject ID', 'Block', 'Trial', 'Condition', 'Load_Condition', 'Emotion_Condition',
                    'Aggregated Performance Score', 'Accuracy', 'Speed', 'Nasa TLX - Anstrengung',
                    'Nasa TLX - Frustration', 'EmojiGrid - Valenz', 'EmojiGrid - Arousal', 'CAAT - Emotion']
    df_complete_trial = df_complete_trial[column_names]
    df_complete_trial['Subject ID'] = id
    df_complete_trial_all = df_complete_trial_all.append(df_complete_trial)


df_complete_trial_all.to_csv(os.path.join(config_analysis.project_directory, 'sourcedata', 'performance', 'performance_trial.csv'), index=False, header=True, decimal=',', sep=';')

# Long format
value_columns = ['Aggregated Performance Score',
                 'Nasa TLX - Anstrengung', 'Nasa TLX - Frustration',
                 'EmojiGrid - Valenz', 'EmojiGrid - Arousal']
id_columns = ['Subject', 'Condition', 'Trial', 'Load_Condition', 'Emotion_Condition']
df_long = pd.melt(df_complete_trial_all, id_vars=id_columns, value_vars=value_columns)
df_long = df_long.loc[df_long['value'].notna()]
df_long.to_csv(os.path.join(config_analysis.project_directory, 'sourcedata', 'performance', 'performance_long.csv'), index=False, decimal=',', sep=';')
