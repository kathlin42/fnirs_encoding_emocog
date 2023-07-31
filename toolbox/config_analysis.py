# =============================================================================
# Set Required Paths and Settings for the Analysis
# =============================================================================

project_directory = 'R:/MIKADO_83820414/!Ergebnisse/mikado_fnirs_2_bids'
fsaverage_directory = 'C:/Users/hirning/mne_data/MNE-sample-data'
time_rest = 30.0
GLM_time_window = 60.0
event_dict = {'Rest': 10,
              'Baseline': 21,
              'LowNeu': 32,
              'LowPos': 33,
              'LowNeg': 34,
              'HighNeu': 42,
              'HighPos': 43,
              'HighNeg': 44}

single_effects = [["LowNeg", "LowPos"],
                  ["LowNeu", "LowNeg"],
                  ["LowNeu", "LowPos"],

                  ["HighNeg", "HighPos"],
                  ["HighNeu", "HighNeg"],
                  ["HighNeu", "HighPos"],

                  ["HighNeg", "LowNeg"],
                  ["HighPos", "LowPos"],
                  ["HighNeu", "LowNeu"]]


interaction_effects = [["HighNeg-HighPos", "LowNeg-LowPos"],
                       ["HighNeu-HighNeg", "LowNeu-LowNeg"],
                       ["HighNeu-HighPos", "LowNeu-LowPos"]]

main_emotion_effects = [['Neg', 'Pos'],
                        ['Neu', 'Neg'],
                        ['Neu', 'Pos']]

main_workload_effects = [['High', 'Low']]

dict_cond_naming = {'HighNeg-HighPos-LowNeg-LowPos': 'Neg-Pos High-Low',
                    'HighNeu-HighNeg-LowNeu-LowNeg': 'Neu-Neg High-Low',
                    'HighNeu-HighPos-LowNeu-LowPos': 'Neu-Pos High-Low',
                    'LowNeg-LowPos': 'Low Neg-Pos',
                    'LowNeu-LowNeg': 'Low Neu-Neg',
                    'LowNeu-LowPos': 'Low Neu-Pos',
                    'HighNeg-HighPos': 'High Neg-Pos',
                    'HighNeu-HighNeg': 'High Neu-Neg',
                    'HighNeu-HighPos': 'High Neu-Pos',
                    'HighNeg-LowNeg': 'Neg High-Low',
                    'HighPos-LowPos': 'Pos High-Low',
                    'HighNeu-LowNeu': 'Neu High-Low'}
