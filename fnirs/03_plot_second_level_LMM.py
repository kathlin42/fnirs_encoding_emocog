# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 18:42:07 2021

@author: hirning
"""
# =============================================================================
# Import required packages
# =============================================================================
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

mne.viz.set_3d_backend("pyvista")  # pyvistaqt
import mne_nirs
import statsmodels.formula.api as smf
from natsort import natsort_keygen
from toolbox import config_analysis
# =============================================================================
# Paths and Variables
# =============================================================================
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'~qgis directory\apps\Qt5\plugins'
os.environ['PATH'] += r';~qgis directory\apps\qgis\bin;~qgis directory\apps\Qt5\bin'

analysis_settings = 'fNIRS_GLM_window_'
data_directory = os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_preproc', analysis_settings + str(config_analysis.GLM_time_window))
save_directory = os.path.join(config_analysis.project_directory, 'derivatives', 'fnirs_encoding', analysis_settings + str(config_analysis.GLM_time_window))

if not os.path.exists("{}".format(save_directory)):
    print('creating path for saving')
    os.makedirs("{}".format(save_directory))

cmap = mpl.cm.seismic
# =============================================================================
# Load Data
# =============================================================================
conditions = ['LowNeu', 'LowPos', 'LowNeg', 'HighNeu', 'HighPos', 'HighNeg']
df_cha = pd.read_csv(os.path.join(data_directory, 'nirs_glm_cha.csv'), sep=';', decimal=',')
df_con = pd.read_csv(os.path.join(data_directory, 'nirs_glm_con.csv'), sep=';', decimal=',')
exemplary_raw_haemo = mne.io.read_raw_fif(os.path.join(data_directory,  "exemplary_raw.fif")).load_data()
exemplary_raw_montage = mne.io.read_raw_snirf(os.path.join(config_analysis.project_directory, 'sourcedata', 'fNIRS', 'VP02', '2021-06-01_001', "2021-06-01_001.snirf"), optode_frame='mri', verbose=True).load_data()
exemplary_raw_haemo.info['bads'] = []
mne.datasets.fetch_fsaverage()

# =============================================================================
# Plot Montage
# =============================================================================
brain = mne.viz.Brain('fsaverage', subjects_dir=os.path.join(config_analysis.fsaverage_directory, 'subjects'), background='w', cortex='0.5')
brain.add_sensors(exemplary_raw_montage.info, trans='fsaverage', fnirs=['channels', 'pairs', 'sources', 'detectors'])
brain.show_view(azimuth=120, elevation=90, distance=450)
brain.save_image(os.path.join(save_directory, 'Brain_visualization_channels_optodes.jpg'))

# =============================================================================
# Plot LMM Results on 3D fsaverage Brain
# =============================================================================
plots = ['contrast_pos_HighLow', 'contrast_neu_HighLow', 'contrast_neg_HighLow',
         'contrast_low_NegPos', 'contrast_low_NeuNeg', 'contrast_low_NeuPos',
         'contrast_high_NegPos', 'contrast_high_NeuNeg', 'contrast_high_NeuPos',

         'contrast_inter_EMO_NegPos', 'contrast_inter_EMO_NeuNeg','contrast_inter_EMO_NeuPos',
         'contrast_inter_WL_NegPos', 'contrast_inter_WL_NeuNeg','contrast_inter_WL_NeuPos',

         'main_emo_NegPos', 'main_emo_NeuNeg', 'main_emo_NeuPos',
         'main_wl_HighLow']

# Cortical Surface Projections for Contrasts
for pick in ['hbo', 'hbr']:
    # pick = 'hbr'
    for cidx, contrast in enumerate(plots):
        # contrast = 'contrast_low_NegNeu'
        con_summary = df_con.loc[df_con['Contrast'] == contrast]
        con_summary = con_summary.loc[con_summary['Chroma'] == pick]
        con_model = smf.mixedlm("effect ~ -1 + ch_name:Chroma", con_summary, groups=con_summary["ID"]).fit(method='nm')
        con_model_df = mne_nirs.statistics.statsmodels_to_results(con_model)
        con_model_df = con_model_df.sort_index()
        # Replace Values with Coefficients calculated in R lmer
        df_con_coef = pd.read_csv(os.path.join(data_directory, 'LMM_coefficients', 'coefficients_' + contrast + '_' + pick + '.csv'))
        df_con_coef = df_con_coef.rename(columns={"Unnamed: 0": "Rowname"})
        df_con_coef['Significant'] = [False if np.min(np.linspace(row_lower, row_upper, 50)) < 0 < np.max(np.linspace(row_lower, row_upper, 50)) else True for row_lower, row_upper in zip(df_con_coef['[0.025'], df_con_coef['0.975]'])]
        df_con_coef = df_con_coef.set_index('Rowname').sort_index()

        con_model_df['Coef.'] = df_con_coef['Estimate'].values
        con_model_df['Std.Err.'] = df_con_coef['Std. Error'].values
        con_model_df['t'] = df_con_coef['t value'].values
        con_model_df['P>|z|'] = df_con_coef['Pr(>|t|)'].values
        con_model_df['[0.025'] = df_con_coef['[0.025'].values
        con_model_df['0.975]'] = df_con_coef['0.975]'].values
        con_model_df['Significant'] = df_con_coef['Significant'].values
        #Bonferroni Correction
        p = con_model_df["P>|z|"].values
        t = np.where(con_model_df['Significant'] == False)[0]
        con_model_df.loc[con_model_df.index.isin(con_model_df.index[t]), 'Coef.'] = 0
        con_model_df = con_model_df.drop(columns=['z'])

        con_model_df = con_model_df.sort_values(by='ch_name', key=natsort_keygen())
        con_model_df['Coef._corr'] = con_model_df['Coef.'] + abs(con_model_df['Coef.']).max()

        for view in ['rostral', 'lateral']:
            if view == 'lateral':
                hemis = ['lh', 'rh']
            else:
                hemis = ['both']
            for hemi in hemis:
                # hemi = 'both'
                brain = mne_nirs.visualisation.plot_glm_surface_projection(
                    exemplary_raw_haemo.copy().pick(picks=pick),
                    statsmodel_df=con_model_df, picks=pick,
                    view=view, hemi=hemi, clim = {'kind' : 'value', 'lims' : (-1.5,0,1.5)},
                    colormap=cmap, colorbar=False, size=(800, 700))
                label = 'HbO' if pick == 'hbo' else 'HbR'
                if 'inter' in contrast:
                    subfolder = 'inter_effects'
                elif 'main' in contrast:
                    subfolder = 'main_effects'
                else:
                    subfolder = 'single_effects'
                if not os.path.exists(os.path.join(save_directory, label, subfolder)):
                    os.makedirs(os.path.join(save_directory, label, subfolder))
                brain.save_image(os.path.join(save_directory, label, subfolder, f'{contrast}_{view}_{hemi}_{pick}.png'))
                brain.close()


fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.02, 1])
cb = mne.viz.plot_brain_colorbar(ax, clim = {'kind':'value', 'lims':(-1.5, 0, 1.5)},
                            colormap=mpl.cm.seismic, orientation='vertical',
                            label='Sign. GLM estimates', bgcolor='0.5')
cb.set_label('Sign. LMM estimates', size = 11, family='Times New Roman', weight = 'bold', loc ='center')
cb.set_ticks(np.linspace(-1.5,1.5,7), size = 11, family='Times New Roman', weight = 'bold')
cb.set_ticklabels(np.linspace(-1.5,1.5,7), size = 11, family='Times New Roman')
plt.show()

fig.savefig(os.path.join(save_directory,'colorbar_vertical.svg'), bbox_inches='tight')

fig = plt.figure()
ax = fig.add_axes([0.05, 0.1, 0.5, 0.02])
cb = mne.viz.plot_brain_colorbar(ax, clim = {'kind':'value', 'lims':(-1.5, 0, 1.5)},
                            colormap=mpl.cm.seismic, orientation='horizontal',
                            label='Sign. GLM estimates', bgcolor='0.5')
cb.set_label('Sign. LMM estimates', size = 11, family='Times New Roman', weight = 'bold', loc ='center')
cb.set_ticks(np.linspace(-1.5,1.5,7), size = 11, family='Times New Roman', weight = 'bold')
cb.set_ticklabels(np.linspace(-1.5,1.5,7), size = 11, family='Times New Roman')
plt.show()

fig.savefig(os.path.join(save_directory,'colorbar_horizontal.svg'), bbox_inches='tight')