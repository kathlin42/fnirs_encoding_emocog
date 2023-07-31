# =============================================================================
# Directories and Imports
# =============================================================================
import os
import pandas as pd
import pingouin as pg
from toolbox import config_analysis

# =============================================================================
# Prepare data
# =============================================================================
df_screening = pd.read_csv(os.path.join(config_analysis.project_directory, 'sourcedata', 'questionnaires', 'MIKADO2_Screening.csv'), sep=';', decimal=',')
df_end = pd.read_csv(os.path.join(config_analysis.project_directory, 'sourcedata', 'questionnaires', 'MIKADO2_End.csv'), sep=';', decimal=',')
df_start = pd.read_csv(os.path.join(config_analysis.project_directory,'sourcedata', 'questionnaires', 'MIKADO2_Start.csv'), sep=';', decimal=',')

subjects = [int(f[-2:])  for f in os.listdir(os.path.join(config_analysis.project_directory, 'sourcedata', 'fnirs')) if not (f.startswith('.') or f.startswith('_'))]
subjects.sort()
df_screening = df_screening.loc[df_screening['Subject ID'].isin(subjects)]
df_end = df_end.loc[df_end['Subject ID'].isin(subjects)]
df_start = df_start.loc[df_start['Subject ID'].isin(subjects)]

# =============================================================================
# BFI-K
# =============================================================================
bfik_cols = df_end.columns[1:22]
bfik_cols_r = df_end.columns[[1, 2, 8, 9, 11, 12, 17, 21]]
bfik_cols_e = df_end.columns[[1, 6, 11, 16]]
bfik_cols_a = df_end.columns[[2, 7, 12, 17]]
bfik_cols_c = df_end.columns[[3, 8, 13, 18]]
bfik_cols_n = df_end.columns[[4, 9, 14, 19]]
bfik_cols_o = df_end.columns[[5, 10, 15, 20, 21]]

# Encoding in scores
df_end[bfik_cols] = df_end[bfik_cols].replace({
    'sehr unzutreffend': '1',
    'eher unzutreffend': '2',
    'weder noch': '3',
    'eher zutreffend': '4',
    'sehr zutreffend': '5'})
# Reverse coded items
df_end[bfik_cols_r] = df_end[bfik_cols_r].replace({
    '1': '5',
    '2': '4',
    '4': '2',
    '5': '1'})
# to numeric
for x in bfik_cols:
    df_end[x] = pd.to_numeric(df_end[x])
# Computation of mean scores
df_end['extraversion'] = df_end[bfik_cols_e].mean(axis=1)
df_end['agreeableness'] = df_end[bfik_cols_a].mean(axis=1)
df_end['conscientiousness '] = df_end[bfik_cols_c].mean(axis=1)
df_end['neuroticism'] = df_end[bfik_cols_n].mean(axis=1)
df_end['openness'] = df_end[bfik_cols_o].mean(axis=1)

alpha_e = pg.cronbach_alpha(data=df_end[bfik_cols_e])
alpha_a = pg.cronbach_alpha(data=df_end[bfik_cols_a])
alpha_c = pg.cronbach_alpha(data=df_end[bfik_cols_c])
alpha_n = pg.cronbach_alpha(data=df_end[bfik_cols_n])
alpha_o = pg.cronbach_alpha(data=df_end[bfik_cols_o])

# =============================================================================
# APSA
# =============================================================================
apsa_cols = df_end.columns[22:43]
# Schwierigkeiten in prospektiver Gedächtnisleistung
apsa_cols_apf1 = df_end.columns[[22, 24, 26, 32, 34, 36, 37, 38, 41]]
# Schwierigkeiten beim Aufrechterhalten der fokussierten Aufmerksamkeitsleistung
apsa_cols_apf2 = df_end.columns[[23, 25, 27, 28, 29, 31, 39, 40, 42]]

# Encoding in scores
df_end[apsa_cols] = df_end[apsa_cols].replace({
    'nie': '0',
    'selten': '1',
    'manchmal': '2',
    'oft': '3',
    'immer': '4'})
# to numeric
for x in apsa_cols:
    df_end[x] = pd.to_numeric(df_end[x])
# Computation of mean scores
df_end['ap-f1'] = df_end[apsa_cols_apf1].mean(axis=1)
df_end['ap-f2'] = df_end[apsa_cols_apf2].mean(axis=1)
df_end['aps20'] = df_end[apsa_cols].mean(axis=1)

alpha_apsa = pg.cronbach_alpha(data=df_end[apsa_cols])
alpha_apf1 = pg.cronbach_alpha(data=df_end[apsa_cols_apf1])
alpha_apf2 = pg.cronbach_alpha(data=df_end[apsa_cols_apf2])


# =============================================================================
# BIS–11
# =============================================================================
bis_cols = df_end.columns[43:73]
bis_cols_r = df_end.columns[[43, 49, 50, 51, 52, 54, 55, 57, 62, 71, 72]]
# Nonplanning Impulsiveness
bis_cols_ni = df_end.columns[[43, 49, 54, 55, 57, 62, 69, 71, 72]]
# Motor Impulsiveness
bis_cols_mi = df_end.columns[[45, 58, 59, 61]]
# Attentional Impulsiveness
bis_cols_ai = df_end.columns[[44, 45, 46, 47, 48, 50, 51, 52, 53, 56, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 70, 71]]

# Encoding in scores
df_end[bis_cols] = df_end[bis_cols].replace({
    'nie/selten': '1',
    'manchmal': '2',
    'oft': '3',
    'fast immer/immer': '4'})
# Reverse coded items
df_end[bis_cols_r] = df_end[bis_cols_r].replace({
    '1': '4',
    '2': '3',
    '3': '2',
    '4': '1'})
# to numeric
for x in bis_cols:
    df_end[x] = pd.to_numeric(df_end[x])
# Computation of mean scores
df_end['bis11_sum'] = df_end[bis_cols].sum(axis=1)
df_end['bis11_mean'] = df_end[bis_cols].mean(axis=1)
df_end['bis-nonplanning_sum'] = df_end[bis_cols_ni].sum(axis=1)
df_end['bis-nonplanning_mean'] = df_end[bis_cols_ni].mean(axis=1)
df_end['bis-motor_sum'] = df_end[bis_cols_mi].sum(axis=1)
df_end['bis-motor_mean'] = df_end[bis_cols_mi].mean(axis=1)
df_end['bis-attention_sum'] = df_end[bis_cols_ai].sum(axis=1)
df_end['bis-attention_mean'] = df_end[bis_cols_ai].mean(axis=1)

alpha_bis11 = pg.cronbach_alpha(data=df_end[bis_cols])
alpha_bis_nonplanning = pg.cronbach_alpha(data=df_end[bis_cols_ni])
alpha_bis_motor = pg.cronbach_alpha(data=df_end[bis_cols_mi])
alpha_bis_attention = pg.cronbach_alpha(data=df_end[bis_cols_ai])

# =============================================================================
# STAI
# =============================================================================
stai_cols = df_end.columns[73:113]
stai_state_cols = df_end.columns[73:93]
stai_trait_cols = df_end.columns[93:113]
stai_cols_r = df_end.columns[[73, 74, 77, 80, 82, 83, 87, 88, 91, 92, 93, 98, 99, 102, 105, 108, 111]]

# Encoding in scores
df_end[stai_state_cols] = df_end[stai_state_cols].replace({
    'überhaupt nicht': '1',
    'ein wenig': '2',
    'ziemlich': '3',
    'sehr': '4'})
df_end[stai_trait_cols] = df_end[stai_trait_cols].replace({
    'fast nie': '1',
    'manchmal': '2',
    'oft': '3',
    'immer': '4'})
# Reverse coded items
df_end[stai_cols_r] = df_end[stai_cols_r].replace({
    '1': '4',
    '2': '3',
    '3': '2',
    '4': '1'})
# to numeric
for x in stai_cols:
    df_end[x] = pd.to_numeric(df_end[x])
# Computation of mean scores
df_end['state_anxiety_sum'] = df_end[stai_state_cols].sum(axis=1)
df_end['state_anxiety_mean'] = df_end[stai_state_cols].mean(axis=1)
df_end['trait_anxiety_sum'] = df_end[stai_trait_cols].sum(axis=1)
df_end['trait_anxiety_mean'] = df_end[stai_trait_cols].mean(axis=1)

alpha_stai_state = pg.cronbach_alpha(data=df_end[stai_state_cols])
alpha_stai_trait = pg.cronbach_alpha(data=df_end[stai_trait_cols])

df_score = df_end.drop(columns=df_end.columns[1:113])

# =============================================================================
# Screening and Start - Preparation
# =============================================================================
# Encoding in scores
df_screening[df_screening.columns[5]] = df_screening[df_screening.columns[5]].replace({
    'Gar nicht': '1',
    'Ein wenig': '2',
    'Gut': '3',
    'Sehr gut': '4'})
df_screening[df_screening.columns[1]] = df_screening[df_screening.columns[1]].replace({
    'weiblich': 'female',
    'männlich': 'male'})
df_screening[df_screening.columns[7]] = df_screening[df_screening.columns[7]].replace({
    'Rechts': 'right',
    'Links': 'left'})
df_screening.columns = ['Subject ID', 'gender', 'age', 'education', 'German level', 'experience computer games', 'job', 'hand', 'alcohol', 'previous participation']
df_start.columns = ['Subject ID', 'valence', 'arousal', 'emotion', 'tiredness', 'motivation']

# =============================================================================
# Merge and Save
# =============================================================================
df_demo = df_screening.merge(df_score, how='outer', on='Subject ID')
df_demo = df_demo.merge(df_start, how='outer', on='Subject ID')
df_demo.to_csv(os.path.join(config_analysis.project_directory, 'sourcedata', 'demographics_scores.csv'), index=False, decimal=',', sep=';')
