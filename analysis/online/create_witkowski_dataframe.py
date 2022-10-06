#!/usr/bin/env python

"""
Brief script to generate the pandas DataFrame used in the online analysis script for data comparison between
experimental data and data published in

Witkowski, M., Cortese, M., Cempini, M., Mellinger, J., Vitiello, N., & Soekadar, S. R. (2014).
Enhancing brain-machine interface (BMI) control of a hand exoskeleton using electrooculography (EOG).
Journal of NeuroEngineering and Rehabilitation, 11(1), 165.
https://doi.org/10.1186/1743-0003-11-165
"""


import pandas as pd

witkowski2014_data_filename = 'data_witkowski2014.csv'

data_witkowski_close_cond1 = [82.3, 82.3, 80.7, 79.1, 77.5, 75.7, 74.7, 73.3, 70, 68.2, 67.4, 64.4, 64.4, 63, 62.7, 62.6, 61.7, 61.5, 61, 60.2, 57.3, 57.3, 56.7, 55.5, 52.8, 51, 49.3, 47.2, 46.8, 41.8]
data_witkowski_relax_cond1 = [56.2, 53.8, 53.8, 46.6, 46.4, 46.2, 45.6, 45.2, 45.1, 44, 43.6, 42.1, 40.3, 38.2, 38, 36.1, 34.6, 31.3, 31.3, 30.5, 28.2, 27.9, 25.2, 24.9, 24.8, 24.8, 21.7, 21.2, 19.5, 17.8]
df_witkowski = pd.DataFrame([dict(Condition='Relax', Feedback=True, close_percentage=value, FilterDescription='Witkowski et al.') for value in data_witkowski_relax_cond1] +
                            [dict(Condition='Close', Feedback=True, close_percentage=value, FilterDescription='Witkowski et al.') for value in data_witkowski_close_cond1]
                             )


df_witkowski.to_csv(witkowski2014_data_filename)