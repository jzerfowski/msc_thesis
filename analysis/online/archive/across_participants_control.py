#!/usr/bin/env python

"""Module documentation goes here"""
import mne
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import itertools
import mne

from matplotlib import pyplot as plt

from analysis.utils_analysis import epochs_get_labels, epochs_selector, epochs_get_conditions, \
    epochs_data_frame_fix_condition
from analysis.utils_plotting import palette, set_context, save_fig
from data_organizer import ExperimentOrganizer

latex_remaps = {'close_percentage': "ClosePercentage", "filter_description": "FilterDescription",}

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
pd.set_option('display.max_rows', 60)

organizer = ExperimentOrganizer()

experiments = organizer.get_experiments_for_analysis('all')
participant_filters = ['analysis_erd_timecourse', 'analysis_online']
participant_filters_descriptions = ['Participants with ERD', 'All participants']

blocks = [2, 3, 4, 5]
# blocks = [1]
feedback = None

def _classify(x):
    if x < 0:
        return 'CLOSE'
    else:
        return 'RELAX'

##
CLOSE_SPEED_PERCENT_PER_SEC: float = 33  # For PacmanWidget (from OPMFeedbackApp.py)
CHANGE_COLOR_PER_SECOND = 0.33  # in [1/s]  # For RelaxFeedbackWidget (from OPMFeedbackApp.py)
tmin, tmax = 0.1, 5.1  # to compensate for the delay of classification
FULL_CLOSE_DURATION = tmax - tmin  # Time it takes for a full close of the pacman/emulated exoskeleton hand

df_full = pd.DataFrame()

for experiment in experiments:
    epochs = experiment.get_epochs_online()
    # epochs = epochs_selector(epochs, blocks=blocks, feedback=feedback)
    epochs_2 = epochs_selector(epochs, blocks=[1, 2, 3, 4, 5], feedback=True)
    epochs_3 = epochs_selector(epochs, blocks=[2, 3, 4, 5], feedback=False)
    epochs = mne.concatenate_epochs([epochs_2, epochs_3])
    sfreq = epochs.info['sfreq']

    # Definitions for successful and failed trial from
    # Enhancing brain-machine interface (BMI) control of a hand exoskeleton using electrooculography (EOG)
    # A grasping motion was defined as successful if the hand exoskeleton closed the hand more than 50% while the green square was displayed.
    n_samples_close_successful = FULL_CLOSE_DURATION * 0.5 * sfreq

    # A violation of the safety criterion was defined as a closing motion that exceeded 25% of a full hand closing while a red square was displayed.
    n_samples_relax_failed = FULL_CLOSE_DURATION * 0.25 * sfreq


    epochs_classified = epochs.copy().apply_function(np.vectorize(_classify), dtype=str, channel_wise=True, picks='all')

    conditions = epochs_get_conditions(epochs_classified)
    data_classified = epochs_classified.get_data(tmin=tmin, tmax=tmax)
    close_percentages = np.sum(data_classified == 'CLOSE', axis=-1).squeeze()/(FULL_CLOSE_DURATION * sfreq) * 100
    close_percentages[close_percentages > 100.0] = 100.0

    classification = scipy.stats.mode(data_classified.squeeze(), axis=1)[0].squeeze()  # Majority classifier

    # success = np.unique(data_classified.squeeze(), return_counts=True, axis=-1)

    # epochs_data_frame_fix_condition

    # arr = np.array([conditions, close_percentages]).T
    df = pd.DataFrame(zip(conditions, close_percentages, classification), columns=['Condition', 'close_percentage', 'classification'])
    df = epochs_data_frame_fix_condition(df)
    df['success'] = df['Condition'] == df['classification']
    df['sfreq'] = sfreq

    for participant_filter in participant_filters:
        df[participant_filter] = experiment.get_info()[participant_filter] == 'X'

    df['participant_id'] = experiment.get_participant_id()
    df_full = pd.concat([df_full, df], ignore_index=True, )
    print(experiment.get_participant_id(), df.groupby(['Condition', 'Block', 'Feedback']).count())


##
df_full['success_exo'] = False
df_full.loc[df_full['Condition'] == 'Close', 'success_exo'] = df_full.loc[df_full['Condition'] == 'Close', 'close_percentage'] > 50
df_full.loc[df_full['Condition'] == 'Relax', 'success_exo'] = df_full.loc[df_full['Condition'] == 'Relax', 'close_percentage'] < 25
df_full.loc[df_full['Condition'] == 'Relax', 'safety_violation'] = df_full.loc[df_full['Condition'] == 'Relax', 'close_percentage'] >= 25
df_full['safety_violation'] = df_full['safety_violation'].fillna(False)

for participant_filter in participant_filters:
    df_full[participant_filter] = df_full[participant_filter].astype(bool)

success_rates_dict = {}
empirical_chance_levels_dict = {}
df_rates_allsets = pd.DataFrame()

for participant_filter, filter_description in zip(participant_filters, participant_filters_descriptions):
    print(f"Statistics for {filter_description}")

    df_subset = df_full[df_full[participant_filter]]
    df_rates = df_subset.groupby(['participant_id', 'Condition', 'Feedback'], as_index=False, dropna=False).mean()

    df_rates['subset'] = df_rates['analysis_erd_timecourse'].map({0.0: "No ERD", 1.0: "ERD"})

    print("Average success close_percentages_dict:")
    df_groups = df_rates.groupby(['Condition', 'Feedback'], dropna=False)
    # print(df_groups.mean().combine(df_groups.std(), lambda x, y: f"{np.mean(x):0.2f}&pm{np.mean(y):0.2f}"))
    print(df_rates.groupby(['Condition', 'Feedback'], dropna=False).mean())


    close_percentages_dict = {}
    success_rates_dict[participant_filter] = {}
    for condition, feedback in itertools.product(['Close', 'Relax'], [False, True]):

        empirical_chance_levels_dict[(participant_filter, feedback)] = df_subset.loc[df_subset['Feedback'] == feedback, 'close_percentage'].mean()

        key = (condition, feedback)
        close_percentages = df_rates.loc[(df_rates['Condition'] == condition) & (df_rates['Feedback'] == feedback), 'close_percentage'].to_numpy()
        close_percentages_dict[key] = close_percentages

        success_rates = df_rates.loc[(df_rates['Condition'] == condition) & (df_rates['Feedback'] == feedback), 'success_exo'].to_numpy()
        success_rates_dict[participant_filter][key] = success_rates

        empirical_chance_level = df_rates.loc[df_rates['Feedback'] == feedback, 'close_percentage'].mean()/100

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

    df_stats = pd.DataFrame()
    for key_a, key_b in itertools.combinations(close_percentages_dict.keys(), 2):
        stat_close_percentage, pvalue_close_percentage = scipy.stats.wilcoxon(close_percentages_dict[key_a], close_percentages_dict[key_b])
        stat_success_rate, pvalue_success_rate = scipy.stats.wilcoxon(success_rates_dict[participant_filter][key_a], success_rates_dict[participant_filter][key_b])
        df_new = pd.DataFrame({
            'pvalue_close_percentage': pvalue_close_percentage,
            'pvalue_success_rate': pvalue_success_rate},
            index=pd.MultiIndex.from_tuples([(key_a, key_b)]))
        df_stats = pd.concat([df_stats, df_new])
        # df_stats.loc[key_a, key_b] =
    print(df_stats.unstack())

    safety_violations = df_rates.loc[df_rates.Condition == 'Relax', 'safety_violation']*100
    print(f"Safety violations: {safety_violations.mean():0.2f}±{safety_violations.std():0.2f}")

    df_rates[['filter', 'filter_description']] = participant_filter, filter_description
    df_rates_allsets = pd.concat([df_rates_allsets, df_rates], ignore_index=True)


# empirical_chance_level_no_feedback = df_full.loc[[dict(Feedback=True)]]
print(f"{empirical_chance_levels_dict=}")

set_context('talk')

##
df_participants_no_erd = df_rates[(df_rates['analysis_online'] == 1.0) & ~(df_rates['analysis_erd_timecourse'] == 1.0)]
df_participants_erd = df_rates[(df_rates['analysis_erd_timecourse'] == 1.0)]
fig, ax = plt.subplots()
sns.stripplot(x=df_participants_no_erd['Condition'], y=df_participants_no_erd['close_percentage'], ax=ax, label='Participants without ERD', color='grey', marker='+')
sns.stripplot(x=df_participants_erd['Condition'], y=df_participants_erd['close_percentage'], ax=ax, label='Participants with ERD', color='blue', marker='+')
##
df_rates['Participants'] = df_rates['analysis_erd_timecourse'].map({1.0: "With ERD", 0.0: "Without ERD"})
df_rates_feedback = df_rates[df_rates['Feedback']]
ax = sns.boxplot(x='Condition', y='close_percentage', data=df_rates_allsets, hue='filter_description')
ax.legend('', frameon=False)
ax.legend_ = None

ax = sns.stripplot(x='Condition', y='close_percentage', data=df_rates_feedback, hue='Participants', palette={'Without ERD': 'black', 'With ERD': 'green'})
ax.set(ylabel='Average Close Percentage')
##
relax_percentage = 36.11
relax_std = 10.85
#
# samples = 1000
# for condition_percentage, condition_std
data_witkowski_close_cond1 = [82.3, 82.3, 80.7, 79.1, 77.5, 75.7, 74.7, 73.3, 70, 68.2, 67.4, 64.4, 64.4, 63, 62.7, 62.6, 61.7, 61.5, 61, 60.2, 57.3, 57.3, 56.7, 55.5, 52.8, 51, 49.3, 47.2, 46.8, 41.8]
data_witkowski_relax_cond1 = [56.2, 53.8, 53.8, 46.6, 46.4, 46.2, 45.6, 45.2, 45.1, 44, 43.6, 42.1, 40.3, 38.2, 38, 36.1, 34.6, 31.3, 31.3, 30.5, 28.2, 27.9, 25.2, 24.9, 24.8, 24.8, 21.7, 21.2, 19.5, 17.8]
df_witkowski = pd.DataFrame([dict(Condition='Relax', Feedback=True, close_percentage=value, filter_description='Witkowski et al.') for value in data_witkowski_relax_cond1] +
                            [dict(Condition='Close', Feedback=True, close_percentage=value, filter_description='Witkowski et al.') for value in data_witkowski_close_cond1]
                             )
df_rates_witkowski = pd.concat([df_rates_allsets, df_witkowski])
df_rates_witkowski = df_rates_witkowski[df_rates_witkowski['Feedback'] == True]

##
# # Plot as used on poster for Biomag 2022
# # order = ['All participants', 'Participants with ERD']
# set_context('poster')
#
# # fig = plt.subplots()
# g = sns.FacetGrid(data=df_rates_witkowski, col='Condition',  hue='Condition', gridspec_kws={"wspace":0.0}, palette=palette)
# g.map(sns.swarmplot, 'filter_description', 'close_percentage', dodge=False, order=['All participants', 'Participants with ERD'], marker='o', zorder=10, size=8)
# g.map(sns.pointplot, 'filter_description', 'close_percentage', capsize=.1, join=False, height=6, aspect=.75, order=['All participants', 'Participants with ERD', "Witkowski et al."], ci=95)
# g.set(ylim=(0, 100), ylabel='Average Close classifications ($\%$)', xlabel='')
# for ax in g.axes.flat:
#     ax.axhline(50, label='Chance level classifier', ls='--', color='k')
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=-15)
#
# g.axes.flat[-1].text(0.98, 50, 'Chance level classifier', transform=ax.get_yaxis_transform(), ha='right', va='bottom', size=plt.rcParams['legend.fontsize'])
#
# g.figure.set_size_inches((20, 13.5))
# g.tight_layout()
# save_fig(g.figure, filename=f"close_classification_rate_poster_biomag", subdir='', basepath_png=r'D:\Thesis\docs\20220831_BIOMAG_Oxford\media')

##
set_context('paper')

def fixed_boxplot(*args, label=None, **kwargs):
    # https://github.com/mwaskom/seaborn/issues/915
    sns.boxplot(*args, **kwargs, labels=[label])

g = sns.FacetGrid(data=df_rates_witkowski.rename(columns=latex_remaps), col='Condition',  hue='Condition', gridspec_kws={"wspace":0.0}, palette=palette)
g.map(sns.swarmplot, latex_remaps['filter_description'], latex_remaps['close_percentage'], dodge=False, order=['All participants', 'Participants with ERD'], marker='X', zorder=10)
g.map(sns.pointplot, latex_remaps['filter_description'], latex_remaps['close_percentage'], capsize=.1, join=False, height=6, aspect=.75, order=['All participants', 'Participants with ERD', "Witkowski et al."], ci=95, markers='D', scale = 0.8)
# g.map(fixed_boxplot, latex_remaps['filter_description'], latex_remaps['close_percentage'], order=['All participants', 'Participants with ERD', "Witkowski et al."], boxprops=dict(alpha=.3))

g.set(ylim=(0, 100), ylabel='Average Close classifications ($\%$)', xlabel='')
for ax in g.axes.flat:
    ax.axhline(50, label='Chance level', ls='--', color='k')
    # ax.axhline(empirical_chance_levels_dict[('analysis_online', True)], label='Emp. chance level', ls=':', color='k')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=-15)


g.axes.flat[-1].text(0.98, 50, 'Chance level', transform=ax.get_yaxis_transform(), ha='right', va='bottom', size=plt.rcParams['legend.fontsize'])
# g.axes.flat[-1].text(0.98, empirical_chance_levels_dict[('analysis_online', True)], 'Emp. chance level', transform=ax.get_yaxis_transform(), ha='right', va='top', size=plt.rcParams['legend.fontsize'])

g.figure.set_size_inches((20, 13.5))
g.tight_layout()
save_fig(g.figure, filename=f"close_classification_rate_biomag", subdir='online_analysis', )

##
set_context('paper')

def fixed_boxplot(*args, label=None, **kwargs):
    # https://github.com/mwaskom/seaborn/issues/915
    sns.boxplot(*args, **kwargs, labels=[label])

g = sns.FacetGrid(data=df_rates_witkowski.rename(columns=latex_remaps), col='Condition',  hue='Condition', gridspec_kws={"wspace":0.0}, palette=palette)
# g.map(sns.scatterplot, latex_remaps['filter_description'], latex_remaps['close_percentage'], marker='X', zorder=10)
g.map(sns.swarmplot, latex_remaps['filter_description'], latex_remaps['close_percentage'], dodge=True, order=['All participants', 'Participants with ERD'], marker='o', edgecolor='none', linewidth=4, zorder=10)
g.map(sns.pointplot, latex_remaps['filter_description'], latex_remaps['close_percentage'], color='black', capsize=.1, join=False, height=6, aspect=.75, order=['All participants', 'Participants with ERD', "Witkowski et al."], ci=95, alpha=0.4, markers='_', scale = 1, zorder=1,)

# g.map(fixed_boxplot, latex_remaps['filter_description'], latex_remaps['close_percentage'], order=['All participants', 'Participants with ERD', "Witkowski et al."], boxprops=dict(alpha=.3))

g.set(ylim=(0, 100), ylabel='Average Close classifications ($\%$)', xlabel='')
for ax in g.axes.flat:
    ax.axhline(50, label='Chance level', ls='--', color='k')
    # ax.axhline(empirical_chance_levels_dict[('analysis_online', True)], label='Emp. chance level', ls=':', color='k')

    ax.set_xticklabels(ax.get_xticklabels(), rotation=-15)
    ax.set(ylim=(10, 90))

g.axes.flat[-1].text(0.98, 50.5, 'Chance level', transform=ax.get_yaxis_transform(), ha='right', va='bottom', size=plt.rcParams['legend.fontsize'])
# g.axes.flat[-1].text(0.98, empirical_chance_levels_dict[('analysis_online', True)]-0.5, 'Emp. chance level', transform=ax.get_yaxis_transform(), ha='right', va='top', size=plt.rcParams['legend.fontsize'])

# g.figure.set_size_inches((18, 13.5))
g.tight_layout()
save_fig(g.figure, filename=f"close_classification_rate", subdir='online_analysis', figsize_pgf=(6.5, 4.5))

##
#
set_context('paper')
# sns.boxplot(data=df_full, x='Condition', y='close_percentage', hue='participant_id')
plt.subplots()
df_rates_feedback = df_rates_allsets[df_rates_allsets.Feedback == True]
# g = sns.catplot(data=df_rates_allsets)


fig, ax = plt.subplots()
ax = sns.scatterplot(data=df_rates_allsets.rename(columns=latex_remaps), x='Condition', y=latex_remaps['close_percentage'], hue=latex_remaps['filter_description'], ax=ax)
# ax = sns.swarmplot(data=df_rates_allsets, x='Condition', y='close_percentage', hue='filter', )
# ax = sns.swarmplot(data=df_rates_feedback, x='Condition', y='close_percentage')
ax.axhline(50, label='Chance level classifier', ls='--', color='k')
ax.set(title=filter_description, ylim=(0, 100), ylabel='Close classifications ($\%$)')
ax.legend()

ax = sns.catplot(data=df_rates_allsets.rename(columns=latex_remaps), x=latex_remaps['filter_description'], hue=latex_remaps['filter_description'], col='Condition', y=latex_remaps['close_percentage'])
ax.set(xlabel='Filter')

fig = plt.gcf()
save_fig(fig, filename=f"close_classification_rate_old", subdir='online_analysis')

## Create Latex table for the document
d = {
    ('Close', 'All'): {'close_percent': 60.6, 'success_rate': "69.7 (p<0.01)"},
    ('Close', 'ERD'): {'close_percent': 69.3, 'success_rate': "83.7 (p<0.01)"},
    ('Relax', 'All'): {'close_percent': 36.7, 'success_rate': "37.9 (p<0.01)"},
    ('Relax', 'ERD'): {'close_percent': 30.6, 'success_rate': "51.0 (p<0.01)"},
    }
s_close_percentages = 'Avg. Close %'
df_latex = pd.DataFrame.from_dict(d, orient='index').unstack().swaplevel(0, 1, axis=1)
df_latex = df_latex.reindex(columns=df_latex.columns.reindex(['All', 'ERD'], level=0)[0])
print(df_latex.style.to_latex())
# df_late