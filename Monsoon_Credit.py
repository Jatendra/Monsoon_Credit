# -*- coding: utf-8 -*-
"""
Created on Mon Jan 08 15:49:50 2018

@author: hp
"""

# Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor as KNR

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,VotingClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.preprocessing import RobustScaler

from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2


# Import data and response

data=pd.read_csv("E:/Monsoon_Data/X_train.csv")
test=pd.read_csv("E:/Monsoon_Data/X_validation.csv")


ydf=pd.read_csv("E:/Monsoon_Data/y_train.csv")
y=ydf[['Dependent_Variable']]

# Preview size of train data & test data 

data.shape
test.shape
ydf.shape
data.info()
y.info()


# Check Data header 

data.head()
y.head()

# Remove unnecessary features at first 

del ydf['Unique_ID']
Id=test['Unique_ID']

# Merge the train test data

full=pd.concat([data,test])
full.index=range(44067)

full.shape

del full['Unique_ID']


# Null Value Estimation

def null_predict(data):
    total=data.isnull().sum().sort_values(ascending=False)    
    percentage=((data.isnull().sum()*100)/data.isnull().count()).sort_values(ascending=False)
    missing=pd.concat([total,percentage],axis=1,keys=['Total','Percent'])
    return missing

    
null_predict(data)
 
## Delete features having most null values

full=full.drop(['N32','N26','N27','N25','N31','N30','N29','N28'],axis=1)

## Overlook distribution of diffrent features

data.describe()

##  Exploratory data analytics

#1 For Categorical features

temp=pd.concat([data,y],axis=1)

sns.distplot(temp['C1'])
sns.barplot(x='C1',y='Dependent_Variable',data=temp)

sns.distplot(temp['C2'])
sns.barplot(x='C2',y='Dependent_Variable',data=temp)

sns.distplot(temp['C3'])
sns.barplot(x='C3',y='Dependent_Variable',data=temp)

sns.distplot(temp['C4'])
sns.barplot(x='C4',y='Dependent_Variable',data=temp)

sns.distplot(temp['C5'])
sns.barplot(x='C5',y='Dependent_Variable',data=temp)

sns.distplot(temp['C6'])
sns.barplot(x='C6',y='Dependent_Variable',data=temp)

sns.distplot(temp['C7'])
sns.barplot(x='C7',y='Dependent_Variable',data=temp)

sns.distplot(temp['C8'])
sns.barplot(x='C8',y='Dependent_Variable',data=temp)


#2 For Continuous variables

cont=data[['N1','N2','N3','N4','N5','N6','N7','N8','N9','N10','N11','N12','N13','N14','N15','N16','N17','N18','N19','N20','N21','N22','N23','N24','N33','N34','N35']]

cont1=cont.dropna()
cont1.shape

index=list(cont1.index)
temp1=pd.concat([cont1,y.loc[index]],axis=1)
temp1.info()

sns.distplot(cont1['N1'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N1',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N1'].max()))


sns.distplot(cont1['N2'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N2',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N2'].max()))


sns.distplot(cont1['N3'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N3',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N3'].max()))

fig, axis1 = plt.subplots(1,1,figsize=(12,4))
average= temp1[["N3", "Dependent_Variable"]].groupby(['N3'],as_index=False).mean()
sns.barplot(x='N3', y='Dependent_Variable', data=average)


sns.distplot(cont1['N4'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N4',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N4'].max()))


sns.distplot(cont1['N5'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N5',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N5'].max()))


sns.distplot(cont1['N6'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N6',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N6'].max()))

fig, axis1 = plt.subplots(1,1,figsize=(12,4))
average= temp1[["N6", "Dependent_Variable"]].groupby(['N6'],as_index=False).mean()
sns.barplot(x='N6', y='Dependent_Variable', data=average)


sns.distplot(cont1['N7'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N7',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N7'].max()))


sns.distplot(cont1['N8'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N8',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N8'].max()))


sns.distplot(cont1['N9'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N9',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N9'].max()))


sns.distplot(cont1['N10'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N10',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N10'].max()))


sns.distplot(cont1['N11'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N11',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N11'].max()))


sns.distplot(cont1['N12'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N12',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N12'].max()))


sns.distplot(cont1['N13'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N13',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N13'].max()))


sns.distplot(cont1['N14'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N14',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N14'].max()))


sns.distplot(cont1['N15'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N15',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N15'].max()))


sns.distplot(cont1['N16'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N16',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N16'].max()))


sns.distplot(cont1['N17'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N17',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N17'].max()))


sns.distplot(cont1['N18'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N18',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N18'].max()))


sns.distplot(cont1['N19'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N19',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N19'].max()))


sns.distplot(cont1['N20'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N20',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N20'].max()))


sns.distplot(cont1['N21'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N21',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N21'].max()))


sns.distplot(cont1['N22'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N22',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N22'].max()))


sns.distplot(cont1['N23'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N23',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N23'].max()))


sns.distplot(cont1['N24'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N24',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N24'].max()))


sns.distplot(cont1['N33'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N33',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N33'].max()))


sns.distplot(cont1['N34'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N34',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N34'].max()))


sns.distplot(cont1['N35'])

facet = sns.FacetGrid(data=temp1, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N35',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp1['N35'].max()))


# Missing Value Imputation By KNN

# Correlated Variables
# N2-->N35,N20
# N4-->N8,N20,N9
# N6-->N1,N19,N21 
# N10-->N21,N14,N12
# N11-->N13
# N13-->N11,N22
# N14-->N21,N10
# N15-->N21,N6
# N17-->N9,N4,N24,N8
# N19-->N6,N1
# N20-->N4,N8
# N21-->N14,N10,N6
# N22-->N13,N4
# N23-->N24
# N35-->N20,N2


data=full[0:33050]

scaler=RobustScaler()

train=scaler.fit_transform(data.loc[data.N23.notnull(),['N24']])
test=full.loc[full.N23.isnull(),['N24']]
ya=data.loc[data.N23.notnull(),['N23']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N23.isnull(),['N23']]=pred

##xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

full.loc[full.N13.isnull(),'N13']=3

full.loc[full.N4.isnull(),'N4']=full.loc[((full['C1']==1)&(full['C2']==0)),'N4'].mean()

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N22.notnull(),['N4','N13']])
test=full.loc[full.N22.isnull(),['N4','N13']]
ya=data.loc[data.N22.notnull(),['N22']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N22.isnull(),['N22']]=pred

##xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N11.notnull(),['N13']])
test=full.loc[full.N11.isnull(),['N13']]
ya=data.loc[data.N11.notnull(),['N11']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N11.isnull(),['N11']]=pred

##xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

full.loc[full.N6.isnull(),'N6']=full['N6'].mean()

full.loc[full.N10.isnull(),'N10']=2

full.loc[full.N14.isnull(),'N14']=2

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N21.notnull(),['N6','N10','N14']])
test=full.loc[full.N21.isnull(),['N6','N10','N14']]
ya=data.loc[data.N21.notnull(),['N21']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N21.isnull(),['N21']]=pred

##xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N15.notnull(),['N6','N21']])
test=full.loc[full.N15.isnull(),['N6','N21']]
ya=data.loc[data.N15.notnull(),['N15']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N15.isnull(),['N15']]=pred

##xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N19.notnull(),['N6','N1']])
test=full.loc[full.N19.isnull(),['N6','N1']]
ya=data.loc[data.N19.notnull(),['N19']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N19.isnull(),['N19']]=pred

##xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N20.notnull(),['N4','N8']])
test=full.loc[full.N20.isnull(),['N4','N8']]
ya=data.loc[data.N20.notnull(),['N20']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N20.isnull(),['N20']]=pred

##xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N17.notnull(),['N4','N8','N9','N24']])
test=full.loc[full.N17.isnull(),['N4','N8','N9','N24']]
ya=data.loc[data.N17.notnull(),['N17']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N17.isnull(),['N17']]=pred

##xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

full.loc[full.N35.isnull(),'N35']=full['N35'].median()

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N2.notnull(),['N35','N20']])
test=full.loc[full.N2.isnull(),['N35','N20']]
ya=data.loc[data.N2.notnull(),['N2']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N2.isnull(),['N2']]=pred

## Delete correalated variables

full=full.drop(['N12','N18','N16','N5','N7','N3'],axis=1)


## Feature Seletion By Backward Selection

data=full[0:33050]

# Create ANOVA F-Values
f_classif(data, y)

# Chi2 values

data2=data[['C2','C3','C4','C5','C6','C7']]
data3 = data2.astype(int)
chi2(data3,y)


data1=data.drop(['N2','C8','N4'],axis=1)
y=np.ravel(y)


## Building Model,Cross Validation

random_state = 2

train_x,test_x,train_y,test_y=train_test_split(data1,y,test_size=.2,random_state=25)

classifier=GradientBoostingClassifier(random_state=random_state)

cv_results=cross_val_score(classifier, train_x, y =train_y, scoring = "accuracy", cv = 4, n_jobs=4, verbose = 1)
cv_results.mean()

## Parameter Tunning

# Gradient boosting tunning


gb_param_grid = {'n_estimators' : [200],
                 'learning_rate': [.1],
                 'max_depth': [3],
                 'max_features' : [0.3],
                 'min_samples_leaf':[150]
                }

gsGBC = GridSearchCV(classifier,param_grid = gb_param_grid, cv=4, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(train_x,train_y)
GBC_best = gsGBC.best_estimator_ 


# Best score
gsGBC.best_score_ 

## Train & Test Score

GBC_best.score(train_x,train_y)
GBC_best.score(test_x,test_y)


kfold = StratifiedKFold(n_splits=4)

##Learning Plots
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


g = plot_learning_curve(GBC_best,"GBM learning curves",train_x,train_y,cv=kfold)


##xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

test=full.loc[33050:]
test=test[['C1','C2','C3','C4','C5','C6','C7','N1','N6','N8','N9','N10','N11','N13','N14','N15','N17','N19','N20','N21','N22','N23','N24','N33','N34','N35']]
result=GBC_best.predict_proba(test)
output=pd.DataFrame([result[i][1] for i in range(11017)],columns=['Class_1_Probability'])
output[:5]
out=pd.concat([Id,output],axis=1)
out.to_csv("E:/Monsoon_Data/Result_Final.csv",index=False)

