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
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import statsmodels.formula.api as smf
os.chdir('..')
from toolbox import (helper_ml, helper_plot, config_analysis)

np.random.seed(42)

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'~qgis directory\apps\Qt5\plugins'
os.environ['PATH'] += r';~qgis directory\apps\qgis\bin;~qgis directory\apps\Qt5\bin'

# =============================================================================
# Directories
# =============================================================================
###############################################################################
# Load Data
###############################################################################
fig_format = ['svg', 'png']
classifier_of_interest = 'LDA'
contrast = 'HighNeg_vs_LowNeg_vs_HighPos_vs_LowPos'
epoch_length = 10
include_silence = 'include_silence' #'_include_silence'
analysis_settings = 'fNIRS_decoding_epoch_length_{}_{}'.format(epoch_length, include_silence)
data_path = os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_decoding', analysis_settings)
save_path = os.path.join(data_path, 'plots')
os.makedirs(save_path, exist_ok=True)

exemplary_raw_haemo = mne.io.read_raw_fif(os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_preproc',analysis_settings, "exemplary_raw.fif")).load_data()
exemplary_raw_haemo.info['bads'] = []
mne.datasets.fetch_fsaverage()
specification = 'full_hbo+hbr'
analysis = 'linear_model_k_best'

save = os.path.join(save_path, analysis + '_' + specification)
os.makedirs(save, exist_ok=True)
chroma = specification.split('_')[-1]
color = '#107869'
average_color = '#1a5653'
dummy_classifier_color = '#a5100c'
feature = 'all'
fsize = 18
score_file = ('scores_with_SFS', '.csv', '_SFS')
estimated_chance_level = helper_ml.estimated_chance_level
# =============================================================================
# Read Data
# =============================================================================
if os.path.exists(os.path.join(save, 'df_boot_SFS.csv')):
    df_boot = pd.read_csv(os.path.join(save, 'df_boot_SFS.csv'),
                          header=0,
                          decimal=',', sep=';')
else:
    df_boot = helper_ml.create_df_boot_SFS_scores(data_path, save, analysis, specification, feature, score_file, contrast)
# rename subj to start at 1
dict_rename_subj = dict(zip(list(df_boot['subj'].unique()), list(
    ['sub-0' + str(subj_num) if len(str(subj_num)) == 1 else 'sub-' + str(subj_num) for subj_num in
     range(1, len(df_boot['subj'].unique()) + 1)])))
df_boot['subj'] = df_boot['subj'].replace(dict_rename_subj)
# add average !!! WATCH OUT - NEED TO BE EXCLUDED WHEN LOOKING AT SUBJ DATA
df_average = df_boot.copy()
df_average['subj'] = 'average'
df_boot = pd.concat((df_boot, df_average))
df_boot = df_boot.reset_index(drop=True)

# =============================================================================
# Analyse effect of k on performance
# =============================================================================
dict_k_performance = {'lower': [], 'mean': [], 'upper': []}
for k in range(1, len([col for col in df_boot if 'k_' in col])+1):
    dict_ci = helper_plot.bootstrapping(
        df_boot.loc[(df_boot['subj'] != 'average') & (df_boot['k'] == k), 'test'].values,
        sample_size=None, numb_iterations=5000, alpha=0.95,
        plot_hist=False, as_dict=True, func='mean')
    for key in dict_ci.keys():
        dict_k_performance[key].append(dict_ci[key])
best_k = np.where(dict_k_performance['mean'] == np.max(dict_k_performance['mean']))[0][0] + 1
df_best_k = df_boot.loc[df_boot['k'] == best_k]
#%%
#Plot Figure
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot(sorted(df_boot['k'].unique()), dict_k_performance['mean'], color=color, label='average accuracy', ls = 'solid')
ax.fill_between(sorted(df_boot['k'].unique()), dict_k_performance['lower'], dict_k_performance['upper'], color=color, alpha=0.2)
ax.set_ylim([0.4, 1.1])
ax.set_xticks([2,4,6,8,10,12,14,16,18,20])
ax.set_xticklabels([2,4,6,8,10,12,14,16,18,20])
ax.xaxis.label.set_size(fsize)
ax.yaxis.label.set_size(fsize)
ax.tick_params(axis='both', which='major', labelsize=(fsize))
fig.suptitle('Average Decoding Performance as a Function of Feature Set Size', fontsize= fsize + 2, fontweight='bold')
fig.supxlabel('Number of Features k', fontsize=fsize, fontweight='bold')
fig.supylabel('Performance [AUC ROC] with SFS-LDA', fontsize=fsize, fontweight='bold')
fig.tight_layout()
plt.show()
for end in fig_format:
    fig.savefig("{}/LDA_SFS_K_Exploration_Lineplot.{}".format(save, end))
plt.clf()

#%%
# =============================================================================
# Analyze best k Classifier per Subject and as Average
# =============================================================================
#Plot Figure
### FOR LDA CLASSIFIERS with SFS PER SUBJECT with optimal K
list_color = [color] * len( sorted(df_best_k['subj'].unique())[1:]) + [average_color]
fig, df_boo = helper_plot.plot_single_trial_bootstrapped_boxplots(
    sorted(df_best_k['subj'].unique())[1:] + ['average'], df_best_k, 'subj', 'fold',
    ['test'],
    boot='mean', boot_size=5000,
    title='Optimized HbO+HbR Features in Four-Class Decoding (k ={})'.format(str(best_k)),
    lst_color=list_color, ncolor_group=1,
    fwr_correction=True,
    x_labels=sorted(df_best_k['subj'].unique())[1:] + ['average'], chance_level=0.25,
    legend= ([Line2D([0], [0], color=list_color[0], lw=3)],
            ['LDA with SFS']),
    n_col=1,
    figsize=(8, 6),
    empirical_chance_level=estimated_chance_level, fs = fsize, axtitle = False)
df_boo.to_csv(os.path.join(save, classifier_of_interest + '_Classification_Participantwise_K_{}.csv'.format(best_k)), header=True,
              index=True, decimal=',', sep=';')
for end in fig_format:
    fig.savefig("{}/LDA_Classification_Participantwise_K_{}.{}".format(save, best_k, end))
plt.clf()
#%%
# =============================================================================
# Analyze Features
# =============================================================================
roi, roi_integer, channel_mappings = helper_ml.get_roi_ch_mappings(project = 'mikado')
counts, actual_channels, counts_ch_per_feat_chroma = helper_ml.count_features(df_best_k, best_k, roi)

#%%
#Plot Figure
group_of_interest = 'roi_hbr_hbo_val'
x = np.asarray(list(counts[group_of_interest].keys()))
y = np.asarray(list(counts[group_of_interest].values()))
order = np.argsort(-y)  # Sort indexes in descending order

x_labels = [label.split(' ')[2].replace('time2max','T2P').replace('peak2peak','P2P')  for label in x[order]]

dict_color = {
    'hbo': dict(zip(['rdlPFC', 'ldlPFC', 'rIFG/OFC', 'lIFG/OFC', 'mdlPFC', 'premotor', 'fronto-temporal', 'others'],
                    ['#7E26A6', '#a6278C', '#b13470', '#bc4357', '#c86554', '#d39865', '#E5C695', "#A9A9A9"])),
    'hbr': dict(zip(['rdlPFC', 'ldlPFC', 'rIFG/OFC', 'lIFG/OFC', 'mdlPFC', 'premotor', 'fronto-temporal', 'others'],
                    ["#115f9a", "#1984c5", "#22a7f0", "#48b5c4", "#76c68f", "#a6d75b", "#c9e52f", "#C0C0C0"]))
}


#Create Legend
legend_names = []
legend_color = []
for chroma in dict_color.keys():
    legend_names = legend_names + [chroma.upper() + ' ' + roi for roi in dict_color[chroma].keys()]
    legend_color = legend_color + [dict_color[chroma][roi] for roi in dict_color[chroma].keys()]

color_lines_legend = []
for col in legend_color:
    color_lines_legend.append(Line2D([0], [0], color=col, lw=3))

color_list = [dict_color[label.split(' ')[1]][label.split(' ')[0]] for label in x[order]]

fig, ax = plt.subplots(1, 1, figsize=(13,5))
ax.bar(x[order], y[order], color=color_list)

ax.xaxis.label.set_size(fsize - 2)
ax.yaxis.label.set_size(fsize)
ax.tick_params(axis='both', which='major', labelsize=(fsize))
ax.set_xticklabels(x_labels, rotation=45, ha= 'right', fontsize = fsize - 4 )
ax.legend(color_lines_legend, legend_names, loc='best', facecolor='white', fontsize='x-large', ncol=2)
fig.suptitle('Feature Analysis of the LDA with SFS (k={})'.format(best_k), fontsize= fsize + 3, fontweight='bold')
fig.supxlabel('Features per Channels clustered in Regions of Interest', fontsize=fsize, fontweight='bold')
fig.supylabel('Count across Participants', fontsize=fsize, fontweight='bold')
fig.tight_layout()
plt.show()
for end in fig_format:
    fig.savefig("{}/Feature_exploration_{}_k_{}.{}".format(save, group_of_interest, best_k, end))
plt.clf()
#%%

# =============================================================================
# Plot Counts across Participants on Inflated Brain
# =============================================================================
#create df_filler
max_n_ch = 0
for key in counts_ch_per_feat_chroma.keys():
    chroma = key.split('_')[0]
    feature = key.split('_')[1]
    if chroma == 'hbo':
        print(feature.upper())

    chs = np.unique(np.array(counts_ch_per_feat_chroma[key]), return_counts=True)[0]
    cts = np.unique(np.array(counts_ch_per_feat_chroma[key]), return_counts=True)[1]
    for cu in cts:
        print(cu)
        if cu > max_n_ch:
            max_n_ch = cu

colormap = {'hbo' :  mpl.colors.LinearSegmentedColormap.from_list("", ["white", '#4A236F']), 'hbr':mpl.colors.LinearSegmentedColormap.from_list("", ["white",  '#060C7F'])}
lims_coefficients = (0, max_n_ch / 2, max_n_ch)
for key in counts_ch_per_feat_chroma.keys():
    chroma = key.split('_')[0]
    feature = key.split('_')[1]
    ch_names = [ch for ch in exemplary_raw_haemo.ch_names if chroma in ch]
    con_summary = pd.DataFrame(
    {'ID': np.repeat(range(1,12), len(ch_names)), 'Contrast': [contrast] * len(np.repeat(range(1,12), len(ch_names))),
     'ch_name': np.repeat(ch_names, len(range(1,12))),
     'Chroma': [chroma] * len(np.repeat(range(1,12), len(ch_names))),
     'coef': [0] * len(np.repeat(range(1,12), len(ch_names)))})
    con_model = smf.mixedlm("coef ~ -1 + ch_name:Chroma", con_summary, groups=con_summary["ID"]).fit(
        method='nm')
    df_con_model = mne_nirs.statistics.statsmodels_to_results(con_model)
    df_con_model['Source'] = [int(ss.split('_')[0]) for ss in
                                    [s.split('S')[1] for s in df_con_model.ch_name]]
    df_con_model['Detector'] = [int(dd.split(' ')[0]) for dd in
                                      [d.split('D')[1] for d in df_con_model.ch_name]]
    df_con_model = df_con_model.sort_values(by=['Source', 'Detector'], ascending=[True, True])

    chs = np.unique(np.array(counts_ch_per_feat_chroma[key]), return_counts=True)[0]
    cts = np.unique(np.array(counts_ch_per_feat_chroma[key]), return_counts=True)[1]
    for ch, count in zip(chs, cts):
        print(ch, 'Count:', count)
        df_con_model.loc[df_con_model['ch_name'] == ch + ' ' + chroma, 'Coef.'] = count
        #if chroma == 'hbo':
        #    df_con_model.loc[df_con_model['ch_name'] == ch + ' ' + chroma, 'Coef.'] = count
        #elif chroma == 'hbr':
        #    df_con_model.loc[df_con_model['ch_name'] == ch + ' ' + chroma, 'Coef.'] = count * -1
    # Cortical Surface Projections for Contrasts
    for view in ['rostral', 'lateral']:
        # view = 'lateral'
        if view == 'lateral':
            hemis = ['lh', 'rh']
        else:
            hemis = ['both']
        for hemi in hemis:
            # hemi = 'both'
            save_name = chroma.upper() + '_' + feature.upper()
            brain = mne_nirs.visualisation.plot_glm_surface_projection(
                exemplary_raw_haemo.copy().pick(picks=chroma),
                statsmodel_df=df_con_model, picks=chroma,
                view=view, hemi=hemi, clim={'kind': 'value', 'lims': lims_coefficients},
                colormap=colormap[chroma], colorbar=False, size=(800, 700))


            brain.save_image(
                "{}/{}_{}_{}_{}.png".format(save, save_name, contrast, view, hemi))
            brain.close()

for chroma in colormap.keys():
    fig, ax = plt.subplots(1, 1, figsize=(1, 6))

    cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical',
                                   cmap= colormap[chroma])
    cb.ax.tick_params(labelsize=20)
    fig.tight_layout()
    plt.show()

    fig.savefig("{}/colorbar_{}_{}.svg".format(save, contrast, chroma), bbox_inches='tight')
    plt.close()