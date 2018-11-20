import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

boston=pd.read_csv('../data/train.csv')
sns.pairplot(boston)

#sns.pairplot(data=boston,hue='medv')

#cols=boston.columns[:-1]
#for col in cols:
#    sns.lmplot(x=col,y='medv',data=boston)
    
#Both rm and lstat seems very correlated.
sns.heatmap(boston.corr())
