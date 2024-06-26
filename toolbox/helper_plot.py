# =============================================================================
# Import packages
# =============================================================================
import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import mne_nirs
import mne
mne.viz.set_3d_backend("pyvista")  # pyvistaqt

from toolbox.helper_bootstrap import bootstrapping
from toolbox import config_analysis
mpl.rc('font',family='Times New Roman', weight = 'bold')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = r'~qgis directory\apps\Qt5\plugins'
os.environ['PATH'] += r';~qgis directory\apps\qgis\bin;~qgis directory\apps\Qt5\bin'
# =============================================================================
# Function to Visualise Bootstrapped Means and CIs
# =============================================================================
# Behavioral Data
def plot_colored_errorbar(lst_groups, df_data, col_group, labels, boot='median',
                          boot_size=2000, title='', lst_color=['#09D21D', '#09D5AD', '#15A3D7'],
                          fwr_correction=True, contrasts=False, n_col=3, figsize=(18, 10), fs=20, reduced_text=True,
                          grouped_colors=False, groupsize=1):
    fig = plt.figure(figsize=figsize, facecolor='white')
    fig.suptitle(title, fontsize=fs + 3, fontweight='bold')
    df_full = pd.DataFrame()
    for i, label in enumerate(labels):

        print(label)
        ax = fig.add_subplot(int(np.ceil(len(labels) / n_col)), n_col, i + 1)
        df_plot = df_data.dropna(subset=[label, col_group], axis=0).copy()
        lst_ala = []
        lst_col = []
        lst_boo = []

        for col in lst_groups:
            vals = df_plot.loc[df_plot[col_group] == col, label].values

            lst_ala.append(vals)
            lst_col.append(col)
            if fwr_correction:
                alpha = 1 - (0.05 / len(lst_groups))
            else:
                alpha = 1 - (0.05)
            if boot == 'mean':
                dict_b = bootstrapping(vals,
                                       numb_iterations=boot_size,
                                       alpha=alpha,
                                       as_dict=True,
                                       func='mean')
                lst_boo.append(dict_b)

        df_boo = pd.DataFrame(lst_boo)

        if grouped_colors:
            ax.errorbar(x=np.array(df_boo.index)[0:groupsize],
                        y=df_boo['mean'].values[0:groupsize],
                        yerr=abs((df_boo.loc[:, ['lower', 'upper']].T - df_boo['mean'].values).values)[:, 0:groupsize],
                        marker='o',
                        ls='',
                        capsize=5,
                        c=lst_color[0])

            if len(lst_groups) > groupsize:
                ax.errorbar(x=np.array(df_boo.index)[groupsize:min(2 * groupsize, len(lst_groups))],
                            y=df_boo['mean'].values[groupsize:min(2 * groupsize, len(lst_groups))],
                            yerr=abs((df_boo.loc[:, ['lower', 'upper']].T - df_boo['mean'].values).values)[:,
                                 groupsize:min(2 * groupsize, len(lst_groups))],
                            marker='o',
                            ls='',
                            capsize=5,
                            c=lst_color[2])
            if len(lst_groups) > 2 * groupsize:
                ax.errorbar(x=np.array(df_boo.index)[min(2 * groupsize, len(lst_groups)):],
                            y=df_boo['mean'].values[min(2 * groupsize, len(lst_groups)):],
                            yerr=abs((df_boo.loc[:, ['lower', 'upper']].T - df_boo['mean'].values).values)[:,
                                 min(2 * groupsize, len(lst_groups)):],
                            marker='o',
                            ls='',
                            capsize=5,
                            c=lst_color[1])
        else:
            ax.errorbar(x=np.array(df_boo.index),
                        y=df_boo['mean'].values,
                        yerr=abs((df_boo.loc[:, ['lower', 'upper']].T - df_boo['mean'].values).values),
                        marker='o',
                        ls='',
                        capsize=5,
                        c="k")
        ax.set_xticks(np.arange(0, len(lst_groups), 1))
        if label not in labels[-n_col:]:
            ax.set_xticklabels('' * len(lst_groups), fontsize=fs - 3, rotation=45, ha='right', c='black')
        elif label in labels[-n_col:]:
            ax.set_xticklabels(lst_groups, fontsize=fs - 3, rotation=45, ha='right', c='black')

            if reduced_text:
                reduced_xticklabels = []
                for xlabel in ax.get_xticklabels():
                    xlabel.set_text(xlabel.get_text().replace(xlabel.get_text(), config_analysis.dict_cond_naming[xlabel.get_text()]))
                    reduced_xticklabels.append(xlabel)
                ax.set_xticklabels(reduced_xticklabels, fontsize=fs - 3, rotation=45, ha='right', c='black')

        ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=fs - 3, c='black')
        # Project Specific Information
        if label in ['arousal', 'valence']:
            ax.set_ylabel('Diff. [Score 0 - 10]', fontsize=fs - 1, c='black', fontweight='bold')
        if label == 'effort':
            ax.set_ylabel('Diff. [Score 0 - 20]', fontsize=fs - 1, c='black', fontweight='bold')
        if label in ['reaction time', 'accuracy']:
            ax.set_ylabel('Diff. [Score 0 - 1]', fontsize=fs - 1, c='black', fontweight='bold')
        ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=fs - 3, c='black', fontweight='bold')
        if contrasts:
            ax.axhline(0, c='black', ls='--')
        ax.set_title(label.replace('_', ' ').title(), fontsize=fs, fontweight='bold', c='black')
        df_boo.index = lst_groups
        df_boo.index.rename(label, inplace=True)
        df_full = df_full.append(df_boo)
    fig.tight_layout()
    return df_full, fig

# Behavioral Data - Subjective Arousal and Valence
def emojigrid_scatter(df, valence, arousal, condition, condition_list, title='', xlabel='', ylabel='',
                      colors=['#e87d72', '#53b74c', '#709bf8', '#E2001A', '#179C7D'], fs = 14):
    print('Plotting ...')
    if not condition_list:
        condition_list = df[condition].unique()
        condition_list.sort()

    figure = plt.figure(figsize=(10, 10))

    for i, c in enumerate(condition_list):
        plt.scatter(x=df.loc[df[condition] == c][valence], y=df.loc[df[condition] == c][arousal], c=colors[i], label=c, s=100)

    plt.ylim(0, 10)
    plt.xlim(0, 10)
    plt.xlabel(xlabel, fontsize = fs, fontweight = 'bold')
    plt.ylabel(ylabel, fontsize = fs, fontweight = 'bold')
    plt.yticks(fontsize=fs - 2, fontweight = 'bold')
    plt.xticks(fontsize=fs - 2, fontweight = 'bold')
    plt.title(title, fontsize = fs + 2, fontweight = 'bold')
    plt.legend(fontsize = fs - 1)

    figure.tight_layout()
    plt.show()
    print('Finished Plotting')
    return figure

def plot_boxes_errorbar(lst_groups, df_data, col_group, labels, boot='median',
                        boot_size=2000, title='', fwr_correction=True, contrasts=False, figsize = (6, 4)):
    fig = plt.figure(figsize=figsize, facecolor='white')
    fig.suptitle(title, fontsize=26, fontweight='bold')
    df_boo_all = pd.DataFrame()
    for i, label in enumerate(labels):

        print(label)
        ax = fig.add_subplot(len(labels), 1, i + 1)
        df_plot = df_data.dropna(subset=[label, col_group], axis=0).copy()

        lst_ala = []
        lst_col = []
        lst_boo = []

        for col in lst_groups:
            vals = df_plot.loc[df_plot[col_group] == col, label].values

            lst_ala.append(vals)
            lst_col.append(col)
            if fwr_correction:
                alpha = 1 - (0.05 / len(lst_groups))
            else:
                alpha = 1 - (0.05)
            if boot == 'mean':
                dict_b = bootstrapping(vals,
                                       numb_iterations=boot_size,
                                       alpha=alpha,
                                       as_dict=True,
                                       func='mean')
                lst_boo.append(dict_b)

        df_boo = pd.DataFrame(lst_boo)

        ax.errorbar(x=np.array(df_boo.index),
                            y=df_boo['mean'].values,
                            yerr=abs((df_boo.loc[:, ['lower', 'upper']].T - df_boo['mean'].values).values),
                            marker='o',
                            ls='',
                            capsize=5,
                            c='grey')

        if contrasts:
            ax.axhline(0, c='black', ls='--')

        ax.plot()
        ax.set_xticks(np.arange(0, len(lst_groups), 1))
        ax.set_xticklabels([''] * len(lst_groups), fontsize=20, fontweight='bold', rotation=45, ha='right',
                           c='black')  #

        ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=20, fontweight='bold', c='black')
        df_boo.index = lst_groups
        df_boo.index.rename(label, inplace=True)
        df_boo_all = df_boo_all.append(df_boo)
    ax.set_xticklabels(lst_groups, fontsize=20, fontweight='bold', rotation=45, ha='right', c='black')  #
    fig.tight_layout()
    return fig, df_boo_all


def plot_single_trial_bootstrapped_boxplots(lst_groups, df_data, col_group, subj_col, labels, boot='median',
                  boot_size=5000, title='', lst_color=None, ncolor_group=None, fwr_correction=True,
                  x_labels=None, chance_level=True, legend = True, n_col = 2, figsize = (10, 10),
                  empirical_chance_level = None, fs = 16, axtitle = True):
    fig = plt.figure(figsize=figsize, facecolor='white')
    fig.suptitle(title, fontsize=fs + 3, fontweight='bold')
    df_full = pd.DataFrame()
    for i, label in enumerate(labels):
        print(label)
        ax = fig.add_subplot(int(np.ceil(len(labels) / n_col)), n_col, i + 1)
        df_plot = df_data.dropna(subset=[label, col_group], axis=0).copy()

        lst_ala = []
        lst_col = []
        lst_boo = []

        for col in lst_groups:
            vals = df_plot.loc[df_plot[col_group] == col, label].values

            lst_ala.append(vals)
            lst_col.append(col)
            if fwr_correction:
                alpha = 1 - (0.05 / len(lst_groups))
            else:
                alpha = 1 - (0.05)
            if boot == 'mean':
                dict_b = bootstrapping(vals,
                                       numb_iterations=boot_size,
                                       alpha=alpha,
                                       as_dict=True,
                                       func='mean')
                lst_boo.append(dict_b)

        df_boo = pd.DataFrame(lst_boo)

        bplot = ax.boxplot(lst_ala,
                           notch=True,
                           usermedians=df_boo['mean'].values,
                           conf_intervals=df_boo.loc[:, ['lower', 'upper']].values,
                           showfliers=False,
                           whis=[5, 95],
                           labels=lst_col, patch_artist=True)

        if lst_color is not None:
            for box, color in zip(bplot['boxes'], lst_color * ncolor_group):
                # box.set(color=color)
                box.set(facecolor=color, alpha=0.5)
        if subj_col is not None:
            individual_means = df_plot.loc[:, [subj_col, col_group, label]].groupby([subj_col, col_group]).mean()[label].reset_index()
            N = df_plot.groupby([col_group]).count().iloc[4::4, 0]
            for i_col, col in enumerate(lst_groups):
                scatter_plot = individual_means.loc[individual_means[col_group] == col][label].values
                ax.scatter([i_col + 1] * len(scatter_plot), scatter_plot, c=(lst_color * ncolor_group)[i_col],
                           edgecolors='grey', s=15)

        if len(x_labels) > 1:
            # ax.set_xticks(ax.get_xticks())
            # for line in [4.5 ,  8.5]:
            #    ax.axvline(line, color = 'lightgrey', linestyle = 'dashed')

            ax.set_xticklabels(x_labels, fontsize=fs, fontweight='bold', rotation=45, ha='center', c='black')  #
        else:
            ax.set_xticks([int((np.max(ax.get_xticks()) - np.min(ax.get_xticks())) / 2)])
            ax.set_xticklabels([''] * len(ax.get_xticks()), fontsize=fs, fontweight='bold', rotation=0, ha='center',
                               c='black', va="center")

            ax.set_xlabel(x_labels[0], fontsize=fs, fontweight='bold', rotation=0, ha='center', va="center", c='black')
        ax.set_ylim(chance_level - 0.1, 1.1)
        ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=fs, fontweight='bold')
        if axtitle != False:
            if axtitle == True:
                ax.set_title(label.upper(), fontsize=fs + 2, fontweight='bold')
            elif title == '':
                ax.set_title(axtitle, fontsize=fs, fontweight='bold')
        if chance_level is not None:
            ax.axhline(chance_level, c='black', ls='--')
        if empirical_chance_level is not None:
            ax.axhline(empirical_chance_level['mean'], c='#a5100c', ls='--')
            ax.axhspan(empirical_chance_level['lower'], empirical_chance_level['upper'], color='#a5100c', alpha=0.2)
        #ax.set_ylim(np.min(ax.get_yticks()), np.max(ax.get_yticks()) + 0.05)
        # ax.set_xticklabels(lst_groups, rotation=45, ha='right')
        df_boo.index = lst_groups
        #df_boo.index.rename(label, inplace=True)
        df_boo = df_boo.reset_index()
        df_boo['Label'] = label
        df_full = pd.concat((df_full, df_boo))
    if legend:
        if legend == True:
            color_lines_legend = []
            for line in enumerate(lst_groups):
                color_lines_legend.append(Line2D([0], [0], color=lst_color[line], lw=3))
        else:
            color_lines_legend = legend[0]
            lst_groups =  legend[1]
            if empirical_chance_level is not None:
                color_lines_legend.append(Line2D([0], [0], color='#a5100c', lw=3, ls='--'))
                lst_groups.append('Empirical Chance')
        ax.legend(color_lines_legend,
                  lst_groups,
                  loc='upper right',
                  facecolor='white',
                  fontsize='xx-large',
                  ncol=2)
    if 'optimized' in title:
        fig.text(0.015, 0.5, "Performance [F1 Weighted] with SFS-LDA", fontsize=fs, fontweight='bold', rotation=90, va='center',
                 ha='center', c='black')
    else:
        fig.text(0.015, 0.5, "Performance [F1 Weighted]", fontsize=fs, fontweight='bold', rotation=90, va='center',ha='center', c='black')
    fig.tight_layout(rect=[0.02, 0, 1, 0.985])
    plt.show()
    return fig, df_full

def plot_weights_3d_brain(df, weights_col, chroma, subj_list, colormaps, lims_coefficients, exemplary_raw_haemo, save_name, save, rescaled, mask):
    for classifier in df['classifier'].unique():
        for con in df['con'].unique():
            df_weights = df.loc[
                (df['classifier'] == classifier) & (df['con'] == con), df.columns != 'classifier']
            df_weights['Source'] = [int(ss.split('_')[0]) for ss in
                                            [s.split('S')[1] for s in df_weights.features]]
            df_weights['Detector'] = [int(dd.split(' ')[0]) for dd in
                                              [d.split('D')[1] for d in df_weights.features]]

            con_summary = pd.DataFrame(
                {'ID': df_weights.subj.values,
                 'Contrast': [con] * len(df_weights.subj.values),
                 'ch_name': [ch.rsplit(chroma, 1)[0] + chroma for ch in df_weights.features],
                 'Chroma': [chroma] * len(df_weights.subj.values),
                 'coef': df_weights[weights_col].values})
            con_model = smf.mixedlm("coef ~ -1 + ch_name:Chroma", con_summary, groups=con_summary["ID"]).fit(
                method='nm')
            df_con_model = mne_nirs.statistics.statsmodels_to_results(con_model)
            for subj in subj_list:
                if subj in df['subj'].unique().tolist():
                    coefficients = df_weights.loc[(df_weights['subj'] == subj)].groupby(['features']).mean(
                        numeric_only=True).reset_index()
                    colormap_key = 'subj_' + chroma
                    # Rescale weighted coefficients to be between -1 and 1
                    if rescaled:
                        max_weighted_coef = coefficients[weights_col].abs().max()
                        coefficients[weights_col] = coefficients[weights_col] / max_weighted_coef
                        lims = (-1, 0, 1)
                    else:
                        lims = lims_coefficients

                elif subj == 'average':
                    coefficients = df_weights.groupby(['features']).mean(
                        numeric_only=True).reset_index()

                    colormap_key = chroma
                    if rescaled:
                        # Rescale weighted coefficients to be between -1 and 1
                        max_weighted_coef = coefficients[weights_col].abs().max()
                        coefficients[weights_col] = coefficients[weights_col] / max_weighted_coef
                        lims = (-1, 0, 1)
                    else:
                        lims = lims_coefficients

                elif subj == 'standard_error':
                    coefficients = df_weights.groupby(['features']).mean(numeric_only=True).reset_index()
                    std = df_weights.loc[:, ['features', weights_col]].groupby(['features']).std(
                        numeric_only=True).values
                    coefficients[weights_col] = std / np.sqrt(len(df['subj'].unique().tolist()))
                    colormap_key = subj
                    if rescaled:
                        # Rescale weighted coefficients to be between -1 and 1
                        max_weighted_coef = coefficients[weights_col].abs().max()
                        coefficients[weights_col] = coefficients[weights_col] / max_weighted_coef
                        lims = (0, 0.1, 0.5)
                    else:
                        lims = lims_coefficients

                elif subj == 'weighted_average':
                    coefficients = df_weights.groupby(['features']).mean(
                        numeric_only=True).reset_index()
                    std = df_weights.loc[:, ['features', weights_col]].groupby(['features']).std(
                        numeric_only=True).values
                    coefficients['SE'] = (std / np.sqrt(len(df['subj'].unique().tolist())))
                    # Calculate weighted coefficients
                    coefficients[weights_col] = coefficients[weights_col] / coefficients['SE']
                    if rescaled:
                        # Rescale weighted coefficients to be between -1 and 1
                        max_weighted_coef = coefficients[weights_col].abs().max()
                        coefficients[weights_col] = coefficients[weights_col] / max_weighted_coef
                        lims = (-1, 0, 1)
                    else:
                        lims = lims_coefficients
                    colormap_key = chroma

                if ('pattern' in weights_col) and (subj != 'weighted_average'):
                    max_weighted_coef = coefficients[weights_col].abs().max()
                    coefficients[weights_col] = coefficients[weights_col] / max_weighted_coef
                    lims = lims_coefficients
                if mask != False:
                    coefficients.loc[~coefficients['features'].isin(mask), weights_col] = 0
                print(subj, len(coefficients), len(df_con_model['Coef.'].values))
                assert len(coefficients) == len(df_con_model['Coef.'].values)

                df_con_model['Coef.'] = coefficients[weights_col].values
                df_con_model['Source'] = coefficients['Source'].values
                df_con_model['Detector'] = coefficients['Detector'].values
                df_con_model_sorted = df_con_model.sort_values(by=['Source', 'Detector'],
                                                               ascending=[True, True]).copy()

                #print(list(df_con_model_sorted["ch_name"].values))
                mne.datasets.fetch_fsaverage()
                # Cortical Surface Projections for Contrasts
                for view in ['rostral', 'lateral']:
                    # view = 'lateral'
                    if view == 'lateral':
                        hemis = ['lh', 'rh']
                        colorbar = False
                    else:
                        hemis = ['both']
                        colorbar = True
                    for hemi in hemis:
                        # hemi = 'both'
                        brain = mne_nirs.visualisation.plot_glm_surface_projection(
                            exemplary_raw_haemo.copy().pick(picks=chroma),
                            statsmodel_df=df_con_model_sorted, picks=chroma,
                            view=view, hemi=hemi, clim={'kind': 'value', 'lims': lims},
                            colormap=colormaps[colormap_key], colorbar=colorbar, size=(800, 700))
                        if subj in df['subj'].unique().tolist():
                            if rescaled:
                                save_brain_plot = os.path.join(save, save_name + '_rescaled', 'subject_level', con)
                            else:
                                save_brain_plot = os.path.join(save, save_name, 'subject_level', con)

                            os.makedirs(save_brain_plot, exist_ok=True)
                            brain.save_image(
                                "{}/sub-{}_{}_{}.png".format(save_brain_plot, str(subj), view, hemi))
                            brain.close()
                        else:
                            if rescaled:
                                save_brain_plot = os.path.join(save, save_name + '_rescaled', subj, con)
                            else:
                                save_brain_plot = os.path.join(save, save_name, subj, con)

                            os.makedirs(save_brain_plot, exist_ok=True)
                            brain.save_image(
                                "{}/{}-{}_{}.png".format(save_brain_plot, str(subj), view, hemi))
                            brain.close()