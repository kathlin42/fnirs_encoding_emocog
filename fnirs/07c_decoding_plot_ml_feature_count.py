"""
Plot feature uses over all subjects
"""

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# %% Load Data
data_path = "df_boot_SFS.csv"
df = pd.read_csv(data_path, sep=";")

# %% Select rows with k = k_max
k_max = 10
df_k = df.loc[df["k"] == k_max]
subj_list = list(df_k["subj"].unique())

# %% Sanity Check if every row per subject is the same for every 100 folds
# for subj in subj_list:
#     sub = df_k.loc[df_k["subj"] == subj]
#
#     for k in range(1, k_max + 1):
#         if len(sub[f"k_{k}"].unique()) > 1:
#             print(subj, k)

# %% Count feature occurrences over all subjects for all k_max
counts = {
    "raw_feat": {},  # As it is in the data
    "hbr_hbo": {"hbr": 0, "hbo": 0},  # Only HbR and HbO
    "channel": {},  # Only Channels S-D
    "val_feat": {},  # P2P, min, max, Slope etc.
    "hbr_hbo_val": {},  # Combination of hbr/hbo and val_feat
    "roi": {},  # Regions of interest (l-dlPFC, r-dlPFC, mid-dlPFC, pre-motor)
    "roi_hbr_hbo_val": {}  # Combination of ROI, hbr/hbo and val feat
}

actual_channels = {
    "l-dlPFC": [],
    "r-dlPFC": [],
    "mid-dlPFC": [],
    "pre-motor": []
}

for subj in subj_list:
    sub = df_k.loc[df_k["subj"] == subj]

    for k in range(1, k_max + 1):

        feat = sub.iloc[0][f"k_{k}"]
        comp = feat.split(" ")

        # Raw feature count
        if feat in counts['raw_feat']:
            counts['raw_feat'][feat] += 1
        else:
            counts['raw_feat'][feat] = 1

        #  hbr and hbo feature count
        if "hbr" in feat:
            counts['hbr_hbo']["hbr"] += 1
        elif "hbo" in feat:
            counts['hbr_hbo']["hbo"] += 1

        # Channel count
        if comp[0] in counts['channel']:
            counts['channel'][comp[0]] += 1
        else:
            counts['channel'][comp[0]] = 1

        # Value Features (min, max, Peak2Peak etc.)
        if comp[2] in counts['val_feat']:
            counts['val_feat'][comp[2]] += 1
        else:
            counts['val_feat'][comp[2]] = 1

        # Combination from value features and hbr/hbo
        key_ = f"{comp[1]} {comp[2]}"
        if key_ in counts['hbr_hbo_val']:
            counts['hbr_hbo_val'][key_] += 1
        else:
            counts['hbr_hbo_val'][key_] = 1

        # Regions of interest
        roi_label = "others"
        for key in roi:
            if comp[0] in roi[key]:
                roi_label = key
                break
        if roi_label in counts['roi']:
            counts['roi'][roi_label] += 1
        else:
            counts['roi'][roi_label] = 1

        # Add channel to the actual channel list (to look which channels are actual available in the dataset)
        if comp[0] not in actual_channels[roi_label]:
            actual_channels[roi_label].append(comp[0])

        # Roi + hbr/hbo + val feats
        key_ = f"{roi_label} {comp[1]} {comp[2]}"
        if key_ in counts['roi_hbr_hbo_val']:
            counts['roi_hbr_hbo_val'][key_] += 1
        else:
            counts['roi_hbr_hbo_val'][key_] = 1

# %% Sanity Check if all sum of counts are equal
# for key in counts:
#     print(sum(counts[key].values()))

# %% Print actual channels
for key in actual_channels:
    print(f"{key}:")
    print(actual_channels[key])
    eeg_channels = []
    for chan in actual_channels[key]:
        ch = chan.split("_")
        for c in ch:
            if channel_mappings[c] not in eeg_channels:
                eeg_channels.append(channel_mappings[c])
    print(eeg_channels)
    print("-----------------")

# %%
sns.set_theme()
sns.set(rc={'figure.figsize': (15.7, 8.27)})
color_palette_list = ["spring, summer", "autumn", "winter", "pink"]

# %%
sns.set_palette("Blues")

# Choose dictionary TODO: Do loop if necessary
count_obj = counts['roi_hbr_hbo_val']

x = np.asarray(list(count_obj.keys()))
y = np.asarray(list(count_obj.values()))
order = np.argsort(-y)  # Sort indexes in descending order

g = sns.barplot(x=x[order], y=y[order])
g.set_xticklabels(g.get_xticklabels(), rotation=45, ha="right")
g.set(xlabel="Features", ylabel="Count", title="Plot")

plt.tight_layout()
# plt.show()
plt.savefig("test.png", bbox_inches='tight')
