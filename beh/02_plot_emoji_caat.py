# =============================================================================
# Directories and Imports
# =============================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
from toolbox import config_analysis, helper_plot

data_directory = os.path.join(config_analysis.project_directory, 'sourcedata', 'performance')
save_directory = os.path.join(config_analysis.project_directory, 'derivatives', 'beh_plots')

if not os.path.exists("{}".format(save_directory)):
    print('creating path for saving')
    os.makedirs("{}".format(save_directory))

# =============================================================================
# Set Plotting Variables and Load Data
# =============================================================================

conditions_emotion = ['Neutral', 'Positive', 'Negative']
conditions_workload = ['Baseline', 'Low', 'High']
fig_format = ['.svg']
fs = 20

df = pd.read_csv(os.path.join(data_directory, 'performance.csv'), sep=';', decimal=',')
df = df.loc[~df['Condition'].isin(['LowSil', 'HighSil'])]

# =============================================================================
# Plotting Emoji Grid
# =============================================================================

fig = helper_plot.emojigrid_scatter(df.loc[df['Load_Condition'] == 'Baseline'], valence='EmojiGrid - Valenz',
                  arousal='EmojiGrid - Arousal',
                  condition='Emotion_Condition', condition_list=False,
                  title='Baseline', xlabel='Valence', ylabel='Arousal',
                  colors=['#709bf8'], fs = fs)

for end in fig_format:
    fig.savefig(os.path.join(save_directory, 'emojigrid_baseline' + end))
plt.close()

fig = helper_plot.emojigrid_scatter(df.loc[df['Load_Condition'] == 'Low'], valence='EmojiGrid - Valenz', arousal='EmojiGrid - Arousal',
                  condition='Emotion_Condition', condition_list=conditions_emotion,
                  title='Low', xlabel='Valence', ylabel='Arousal',
                  colors=['#b1e641', '#53b74c', '#087a00'], fs = fs)

for end in fig_format:
    fig.savefig(os.path.join(save_directory, 'emojigrid_low_workload' + end))
plt.close()

fig = helper_plot.emojigrid_scatter(df.loc[df['Load_Condition'] == 'High'], valence='EmojiGrid - Valenz', arousal='EmojiGrid - Arousal',
                  condition='Emotion_Condition', condition_list=conditions_emotion,
                  title='High', xlabel='Valence', ylabel='Arousal',
                  colors=['#f3bab4', '#e87d72', '#dd4030'], fs = fs)
for end in fig_format:
    fig.savefig(os.path.join(save_directory, 'emojigrid_high_workload' + end))
plt.close()

# =============================================================================
# Create DataFrame for CAAT
# =============================================================================

frequencies = df.groupby(['Load_Condition', 'Emotion_Condition'], dropna=False)['CAAT - Emotion'].value_counts()
#frequencies.to_csv(os.path.join(save_directory, 'frequencies_caat.csv'), decimal=',', sep=';')
