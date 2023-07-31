# =============================================================================
# Import packages
# =============================================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from toolbox.helper_bootstrap import bootstrapping
from toolbox import config_analysis
mpl.rc('font',family='Times New Roman', weight = 'bold')

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