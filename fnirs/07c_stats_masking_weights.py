# =============================================================================
# fNIRS Classification - Plot Results
# Use standard features and simple linear models
# with nested cross-validation per subject
# combine cross_val with Grid_Search_CV with Sequential Feature Selection
# =============================================================================

# import necessary modules
import os
import numpy as np
import pandas as pd
import mne
import mne_nirs
mne.viz.set_3d_backend("pyvista")  # pyvistaqt
import matplotlib
os.chdir('..')
from toolbox import (helper_ml, helper_plot, config_analysis)
import statsmodels.formula.api as smf
import matplotlib as mpl
import pickle
np.random.seed(42)
from tqdm import tqdm
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'~qgis directory\apps\Qt5\plugins'
os.environ['PATH'] += r';~qgis directory\apps\qgis\bin;~qgis directory\apps\Qt5\bin'#

# =============================================================================
# Paths and Variables
# =============================================================================
epoch_length = 10
contrast = 'HighNeg_vs_LowNeg_vs_HighPos_vs_LowPos'
contrast_list =['HighNeg', 'LowNeg', 'HighPos', 'LowPos']
include_silence = 'include_silence' #'_include_silence'
analysis_settings = 'fNIRS_decoding_epoch_length_{}_{}'.format(epoch_length, include_silence)
ROI = 'full'
classifier_of_interest = 'LDA'
feature_of_interest = 'hbr_max'
decoding_path = os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_decoding', analysis_settings)
rescaled = True
###############################################################################
# Load Data
###############################################################################
#Load epochs for all subjects
exemplary_raw_haemo = mne.io.read_raw_fif(os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_preproc',analysis_settings, "exemplary_raw.fif")).load_data()
exemplary_raw_haemo.info['bads'] = []
df_epochs = pd.read_csv(os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_preproc', analysis_settings, 'df_epochs.csv'), decimal = ',', sep = ';', header = 0)
# Extract Features and create group variable for features
labels, df_features, features_dict = helper_ml.feature_extraction(df_epochs, feature_of_interest.split('_')[0],
                                                                              'epoch',
                                                                              'ID', [feature_of_interest.split('_')[1]],
                                                                              "condition")
df_features['condition'] = labels
df_features = df_features.groupby(['ID', 'condition']).mean(numeric_only = True).drop('epoch', axis = 1).reset_index()
df_stats = pd.DataFrame()
for feat in tqdm([col for col in df_features.columns if col not in ['ID', 'condition']]):
    for subj in df_features['ID']:
        df_subj = df_features.loc[(df_features['ID'] == subj)  & (df_features['condition'].isin(contrast_list)), ['condition', feat]]
        c1 = df_subj.loc[df_subj['condition'] == contrast_list[0], feat].values
        c2 = df_subj.loc[df_subj['condition'] == contrast_list[1], feat].values
        c3 = df_subj.loc[df_subj['condition'] == contrast_list[2], feat].values
        c4 = df_subj.loc[df_subj['condition'] == contrast_list[3], feat].values
        contrast_val = ((c1 - c2) - (c3 - c4))
        df_stats_subj = pd.DataFrame({'ID': [subj], 'feature': [feat], 'contrast': contrast_val})
        df_stats = pd.concat((df_stats, df_stats_subj))
df_stats['contrast'] = pd.to_numeric(df_stats['contrast'])

results = pd.DataFrame({'feature': df_stats['feature'].unique()})
results['mean'] = np.nan
results['upper'] = np.nan
results['lower'] = np.nan
results['sig'] = np.nan
for feat in tqdm(df_stats['feature'].unique()):
    dict_ci = helper_plot.bootstrapping(df_stats.loc[(df_stats['feature'] ==feat), 'contrast'].values,
                                        sample_size=None, numb_iterations=5000, alpha=1 - (0.05 /len(df_stats['feature'].unique())),
                                        plot_hist=False, as_dict=True, func='mean')
    results.loc[(results['feature'] == feat), 'mean']  = dict_ci['mean']
    results.loc[(results['feature'] == feat), 'upper']  = dict_ci['upper']
    results.loc[(results['feature'] == feat), 'lower']  = dict_ci['lower']

    if np.min(np.linspace(dict_ci['lower'], dict_ci['upper'], 50)) < 0 < np.max(np.linspace(dict_ci['lower'], dict_ci['upper'], 50)):
        results.loc[(results['feature'] == feat), 'sig'] = False
    else:
        results.loc[(results['feature'] == feat), 'sig'] = True
sig_channels = results.loc[results['sig'], 'feature'].to_list()
results.to_csv(os.path.join(decoding_path, 'plots', 'bootstrapped_stats.csv'), sep = ';', decimal=',', header = True, index = False)
# =============================================================================
# define necessary variables
# =============================================================================


channel_masks = pd.read_pickle(os.path.join(decoding_path, 'plots', 'linear_model_k_best' + '_' + 'full_hbo+hbr', 'channel_masks.pickle'))[feature_of_interest]['channels']
counts = pd.read_pickle(os.path.join(decoding_path, 'plots', 'linear_model_k_best' + '_' + 'full_hbo+hbr', 'channel_masks.pickle'))[feature_of_interest]['counts']
ch_names = [ch for ch in exemplary_raw_haemo.ch_names if feature_of_interest.split('_')[0] in ch]

con_summary = pd.DataFrame(
    {'ID': np.repeat(range(1,12), len(ch_names)), 'Contrast': [contrast] * len(np.repeat(range(1,12), len(ch_names))),
     'ch_name': np.repeat(ch_names, len(range(1,12))),
     'Chroma': [feature_of_interest.split('_')[0]] * len(np.repeat(range(1,12), len(ch_names))),
     'coef': [0] * len(np.repeat(range(1,12), len(ch_names)))})
con_model = smf.mixedlm("coef ~ -1 + ch_name:Chroma", con_summary, groups=con_summary["ID"]).fit(
    method='nm')
df_con_model = mne_nirs.statistics.statsmodels_to_results(con_model)
df_con_model['Source'] = [int(ss.split('_')[0]) for ss in
                                [s.split('S')[1] for s in df_con_model.ch_name]]
df_con_model['Detector'] = [int(dd.split(' ')[0]) for dd in
                                  [d.split('D')[1] for d in df_con_model.ch_name]]
df_con_model = df_con_model.sort_values(by=['Source', 'Detector'], ascending=[True, True])
mne.datasets.fetch_fsaverage()
for ch, count in zip(channel_masks, counts):
    print(ch, 'Count:', count)
    df_con_model.loc[df_con_model['ch_name'] == ch + ' ' + feature_of_interest.split('_')[0], 'Coef.'] = count
    # if chroma == 'hbo':
    #    df_con_model.loc[df_con_model['ch_name'] == ch + ' ' + chroma, 'Coef.'] = count
    # elif chroma == 'hbr':
    #    df_con_model.loc[df_con_model['ch_name'] == ch + ' ' + chroma, 'Coef.'] = count * -1
# Cortical Surface Projections for Contrasts
lims_coefficients = (0, np.max(counts) / 2, np.max(counts))
for view in ['rostral', 'lateral']:
    # view = 'lateral'
    if view == 'lateral':
        hemis = ['lh', 'rh']
    else:
        hemis = ['both']
    for hemi in hemis:
        # hemi = 'both'
        brain = mne_nirs.visualisation.plot_glm_surface_projection(
            exemplary_raw_haemo.copy().pick(picks=feature_of_interest.split('_')[0]),
            statsmodel_df=df_con_model, picks=feature_of_interest.split('_')[0],
            view=view, hemi=hemi, clim={'kind': 'value', 'lims': lims_coefficients},
            colormap=mpl.colors.LinearSegmentedColormap.from_list("", ["white",  '#060C7F']), colorbar=False, size=(800, 700))
        os.makedirs(os.path.join(decoding_path, 'plots', 'counting_' + feature_of_interest), exist_ok=True)
        brain.save_image(
            "{}/{}_{}_{}.png".format(os.path.join(decoding_path, 'plots', 'counting_' + feature_of_interest), contrast, view, hemi))
        brain.close()


channel_masks = [ch + ' ' + feature_of_interest.replace('_', ' ') for ch in channel_masks]
# Calculate similarity percentage
mask_similarity = (len(set(channel_masks).intersection(set(sig_channels))) / len(channel_masks)) * 100
sig_similarity = (len(set(channel_masks).intersection(set(sig_channels))) / len(sig_channels)) * 100
print(f"Similarity relative to mask list: {mask_similarity:.2f}%")
print(f"Similarity relative to sig list: {sig_similarity:.2f}%")

#################################################################################

df_coef = pd.read_csv(os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_decoding', analysis_settings, 'plots', 'topographical_analysis_interaction_' + feature_of_interest.split('_')[0], feature_of_interest.split('_')[1], 'df_coef.csv'),
                      header=0, decimal=',', sep=';')
df_coef['features'] = [f[:-1] if f[-1] == ' ' else f for f in df_coef['features']]
set(df_coef['features'].unique()) - set(channel_masks)
set(df_coef['features'].unique()) - set(channel_masks)

subj_list_plotting = ['weighted_average']
colormaps = {'hbr': matplotlib.cm.RdBu_r, 'subj_hbr' : matplotlib.cm.PiYG, 'hbo': matplotlib.cm.RdBu_r, 'subj_hbo' : matplotlib.cm.PiYG_r, 'standard_error': matplotlib.cm.Reds}
for mask_name, mask in [ ('SFS_', channel_masks), ('bootstrapped_', sig_channels), ('no_', False),]: #,

    save = os.path.join(decoding_path, 'plots', mask_name + 'masked_' + feature_of_interest)
    os.makedirs(save, exist_ok=True)
    lims_coefficients = (-1, 0, 1)
    helper_plot.plot_weights_3d_brain(df_coef, 'coef', feature_of_interest.split('_')[0], subj_list_plotting,
                                  colormaps, lims_coefficients, exemplary_raw_haemo,
                                  feature_of_interest.split('_')[1].upper().replace(' ', '_'), save, rescaled=rescaled, mask = mask)

