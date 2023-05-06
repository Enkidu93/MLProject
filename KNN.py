#imports
import math
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import missingno as msno

import matplotlib.pyplot as plt
np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.4f}'.format})

import matplotlib.pyplot as plt
# %matplotlib inline
warnings.filterwarnings("ignore")

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest,f_regression,mutual_info_regression
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error,explained_variance_score

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV
from numpy import arange


from matplotlib import pyplot

# %matplotlib inline
warnings.filterwarnings('ignore')


#getting data, showing preview of it
df = pd.read_csv('acs2017_census_tract_data.csv')
df.head()

#drop data with zero total population
df = df.loc[df['TotalPop'] > 0]

#hmm, we have some NaN values. Lets remove these rows from the table as well
df = df.dropna()
df[["State","County","TractId","Income", 'TotalPop']].sort_values(by = "Income", ascending = False)

#drop some useless columns
df= df.drop(['TractId', 'State', 'County','IncomeErr','IncomePerCap','IncomePerCapErr'], axis=1)
df.head()

#normalization
to_percent = ['Men','Women','VotingAgeCitizen','Employed']
df[to_percent] = df[to_percent].div(df["TotalPop"], axis="index")*100


#split data into test and validation set
X= df.drop(["Income"], axis=1)
y= df.Income
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from time import time

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import explained_variance_score,mean_absolute_error,r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

#making pipeline to predict using 95% variance
pca95 = PCA(.95)
pipe95 = Pipeline(steps=[
             ("scaler", StandardScaler()),
             ("imputer", SimpleImputer()),
             ("pca", pca95)])
X_train= pipe95.fit_transform(X_train)
X_test= pipe95.transform(X_test)
from sklearn.metrics import explained_variance_score,mean_absolute_error,r2_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

model=  KNeighborsRegressor(8),


start = time()
model.fit(X_train, y_train)
train_time = time() - start
start = time()
y_pred = model.predict(X_test)
predict_time = time()-start    
print(model)
print("\tTraining time: %0.3fs" % train_time)
print("\tPrediction time: %0.3fs" % predict_time)
print("\tExplained variance:", explained_variance_score(y_test, y_pred))
print("\tMean absolute error:", mean_absolute_error(y_test, y_pred))
print("\tR2 score:", r2_score(y_test, y_pred))
print()
def predict():
    return y_pred