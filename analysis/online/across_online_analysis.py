#!/usr/bin/env python

"""
Online Analysis
Module to analyze the across-participants performance of the classification pipeline
"""

import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import itertools
import mne

from matplotlib import pyplot as plt

from utils_analysis import epochs_selector, epochs_get_conditions, epochs_data_frame_fix_condition
from utils_plotting import palette, set_context, save_fig
from data_organizer import ExperimentOrganizer

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 60)

organizer = ExperimentOrganizer()

experiments = organizer.get_experiments_for_analysis('all')
participant_filters = ['analysis_online_with_erd', 'analysis_online']
participant_filters_descriptions = ['Participants with ERD', 'All participants']

def _classify(x):
    if x < 0:
        return 'CLOSE'
    else:
        return 'RELAX'


##
tmin, tmax = 0.1, 5.1  # to compensate for the delay of classification (at 10 Hz)
FULL_CLOSE_DURATION = 5  # It takes 5 seconds for a full close of the pacman/emulated exoskeleton hand

df_full = pd.DataFrame()

for experiment in experiments:
    epochs = experiment.get_epochs_online()
    sfreq = epochs.info['sfreq']

    epochs_2 = epochs_selector(epochs, blocks=[1, 2, 3, 4, 5], feedback=True)
    epochs_3 = epochs_selector(epochs, blocks=[2, 3, 4, 5], feedback=False)
    epochs = mne.concatenate_epochs([epochs_2, epochs_3])

    epochs_classified = epochs.copy().apply_function(np.vectorize(_classify), dtype=str, channel_wise=True, picks='all')

    conditions = epochs_get_conditions(epochs_classified)
    data_classified = epochs_classified.get_data(tmin=tmin, tmax=tmax)
    close_percentages = np.sum(data_classified == 'CLOSE', axis=-1).squeeze()/(FULL_CLOSE_DURATION * sfreq) * 100
    close_percentages[close_percentages > 100.0] = 100.0

    # Create a DataFrame containing one row for each trial
    df = pd.DataFrame(zip(conditions, close_percentages), columns=['Condition', 'close_percentage'])

    df = epochs_data_frame_fix_condition(df)
    df['sfreq'] = sfreq
    df['participant_id'] = experiment.get_participant_id()

    for participant_filter in participant_filters:
        df[participant_filter] = experiment.get_info()[participant_filter] == 'X'

    df_full = pd.concat([df_full, df], ignore_index=True, )


## Post-processing after creating the dataframe
# Compute the success rates and the safety violation rates
df_full['success_exo'] = False
df_full.loc[df_full['Condition'] == 'Close', 'success_exo'] = df_full.loc[df_full['Condition'] == 'Close', 'close_percentage'] > 50
df_full.loc[df_full['Condition'] == 'Relax', 'success_exo'] = df_full.loc[df_full['Condition'] == 'Relax', 'close_percentage'] < 25
df_full.loc[df_full['Condition'] == 'Relax', 'safety_violation'] = df_full.loc[df_full['Condition'] == 'Relax', 'close_percentage'] >= 25
df_full['safety_violation'] = df_full['safety_violation'].fillna(False)

##
for participant_filter in participant_filters:
    df_full[participant_filter] = df_full[participant_filter].astype(bool)

## Compute a table with all participants and their average success rates
df_success_rate = df_full[df_full['Feedback'] == True].groupby(['participant_id']).mean().reset_index()[['participant_id', 'analysis_online_with_erd', 'analysis_online']]


##
empirical_chance_levels_dict = {}
df_rates_allsets = pd.DataFrame()

# Compute the statistics for all defined participant filters
for participant_filter, filter_description in zip(participant_filters, participant_filters_descriptions):
    print(f"Computing statistics for {filter_description} ({participant_filter})")

    df_filtered = df_full[df_full[participant_filter] == True]

    df_rates = df_filtered.groupby(['participant_id', 'Condition', 'Feedback', 'Block'], as_index=False, dropna=False).mean()
    df_rates['success_exo_percent'] = df_rates['success_exo'] * 100
    df_rates[['filter', 'FilterDescription']] = participant_filter, filter_description
    df_rates_allsets = pd.concat([df_rates_allsets, df_rates], ignore_index=True)

    # print("Average success close_percentages_dict:")
    print(df_rates.groupby(['Condition', 'Feedback'], dropna=False).mean())

    print(f"Overall average success rate: {df_rates.loc[df_rates['Feedback'] == True, 'success_exo_percent'].mean()}")


    close_percentages_dict = {}
    for condition, feedback in itertools.product(['Close', 'Relax'], [False, True]):
        key = (condition, feedback)

        # Compute the empirical chance levels
        empirical_chance_level = df_filtered.loc[(df_filtered['Feedback'] == feedback), 'close_percentage'].mean()/100
        empirical_chance_levels_dict[(participant_filter, feedback)] = empirical_chance_level

        df_rates_condition = df_filtered.loc[(df_filtered['Condition'] == condition) & (df_filtered['Feedback'] == feedback)]

        success_rates = df_filtered.loc[(df_filtered['Condition'] == condition) & (df_filtered['Feedback'] == feedback)].groupby(['participant_id', 'Condition', 'Feedback']).mean()['success_exo'].to_numpy()

        # Compute the success rates of a chance-level classifier
        if condition == 'Close':
            chance_classifier_hyp = 1-scipy.stats.binom.cdf(25, 50, 0.5)
            empirical_classifier_hyp = 1-scipy.stats.binom.cdf(25, 50, empirical_chance_level)
        elif condition =='Relax':
            chance_classifier_hyp = scipy.stats.binom.cdf(12, 50, 0.5)
            empirical_classifier_hyp = scipy.stats.binom.cdf(12, 50, empirical_chance_level)


        stat_chance, pvalue_chance = scipy.stats.wilcoxon(success_rates - chance_classifier_hyp)
        stat_empirical, pvalue_empirical = scipy.stats.wilcoxon(success_rates - empirical_classifier_hyp)
        print(f"{condition=}, {feedback=}: Avg success rate={np.mean(success_rates*100):0.2f}±{np.std(success_rates*100):0.2f}%\n"
              f"Chance level classifier: H_0 success rate={chance_classifier_hyp*100:0.4}%. {stat_chance=}, {pvalue_chance=:0.4}\n"
              f"Empirical chance classifier ({empirical_chance_level:0.3}): H_0 success rate={empirical_classifier_hyp*100:0.4}%. {stat_empirical=}, {pvalue_empirical=:0.4}")

    safety_violations = df_rates.loc[df_rates.Condition == 'Relax', 'safety_violation']*100
    print(f"Safety violations: {safety_violations.mean():0.2f}±{safety_violations.std():0.2f}")


print(f"{empirical_chance_levels_dict=}")

df_participants_no_erd = df_rates[(df_rates['analysis_online'] == 1.0) & ~(df_rates['analysis_online_with_erd'] == 1.0)]
df_participants_erd = df_rates[(df_rates['analysis_online_with_erd'] == 1.0)]


## Import Witkowski data
df_witkowski = pd.read_csv('data_witkowski2014.csv')

df_rates_witkowski = pd.concat([df_rates_allsets.groupby(['participant_id', 'Condition', 'Feedback', 'FilterDescription']).mean().reset_index(), df_witkowski])
df_rates_witkowski = df_rates_witkowski[df_rates_witkowski['Feedback'] == True]

##
set_context('paper')

g = sns.FacetGrid(data=df_rates_witkowski, col='Condition',  hue='Condition', gridspec_kws={"wspace":0.0}, palette=palette)
g.map(sns.swarmplot, 'FilterDescription', 'close_percentage', dodge=True, order=['All participants', 'Participants with ERD'], marker='o', edgecolor='none', linewidth=4, zorder=10)
g.map(sns.pointplot, 'FilterDescription', 'close_percentage', color='black', capsize=.1, join=False, height=6, aspect=.75, order=['All participants', 'Participants with ERD', "Witkowski et al."], ci=95, alpha=0.4, markers='_', scale = 1, zorder=1,)


g.set(ylim=(0, 100), ylabel='Average Close rate ($\%$)', xlabel='')
for ax in g.axes.flat:
    ax.axhline(50, label='Chance level', ls='--', color='k')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=-15)
    ax.set(ylim=(10, 90))

g.axes.flat[-1].text(0.98, 50.5, 'Chance level', transform=ax.get_yaxis_transform(), ha='right', va='bottom', size=plt.rcParams['legend.fontsize'])
g.tight_layout()
save_fig(g.figure, filename=f"close_classification_rate", subdir='online_analysis', figsize_pgf=(None, 4.5))


##
# Plot success_rate based on block
df_analysis_blocks = df_participants_erd.loc[df_participants_erd['Feedback'] == True]
df_blocks_mean = df_analysis_blocks.groupby(['Block', 'participant_id']).mean().reset_index()
# df_blocks_mean['Block'] = df_blocks_mean['Block'].astype(np.int32)
blocks = np.arange(1, 6)

import statsmodels.formula.api as smf

model_close = smf.mixedlm('success_exo_percent ~ Block', groups='participant_id', data=df_analysis_blocks[df_analysis_blocks['Condition']=='Close']).fit()
model_relax = smf.mixedlm('success_exo_percent ~ Block', groups='participant_id', data=df_analysis_blocks[df_analysis_blocks['Condition']=='Relax']).fit()
model_avg = smf.mixedlm('success_exo_percent ~ Block', groups='participant_id', data=df_blocks_mean).fit()


print("Model for Close condition", model_close.summary(), end='\n\n')
print("Model for Relax condition", model_relax.summary(), end='\n\n')
print("Model for combined success rate", model_avg.summary(), end='\n\n')

##
print("Block statistics: ")
print(df_analysis_blocks.groupby([ 'Condition', 'Block']).mean()['success_exo_percent'])

##
fig, ax = plt.subplots()
sns.pointplot(data=df_analysis_blocks, x='Block', y='success_exo_percent', hue='Condition', palette=palette, orient='v', dodge=True, n_boot=1000, seed=42)
sns.lineplot(x=blocks-1, y = model_close.params['Intercept'] + model_close.params['Block'] * blocks, color=palette['Close'], linestyle='--')
sns.lineplot(x=blocks-1, y = model_relax.params['Intercept'] + model_relax.params['Block'] * blocks, color=palette['Relax'], linestyle='--')

# Compute overall stats (subtract -1 from block because of plotting particularities from seaborn
df_blocks_mean_plot = df_blocks_mean.copy()
df_blocks_mean_plot['Block'] = df_blocks_mean_plot['Block']-1
sns.scatterplot(data=df_blocks_mean_plot.groupby(['Block']).mean(), x='Block', y='success_exo_percent', color='black', ci=None, label='Combined', s=50, zorder=10)
sns.lineplot(data=df_blocks_mean_plot.groupby(['Block']).mean(), x='Block', y='success_exo_percent', color='black', ci=None, zorder=10)
sns.lineplot(x=blocks-1, y = model_avg.params['Intercept'] + model_avg.params['Block'] * blocks, color='black', linestyle='--', zorder=10)

ax.set(xlabel='Block', ylabel='Success rate (\%)')

save_fig(fig, filename=f"success_rate_block", subdir='online_analysis', figsize_pgf=(None, 4.5))