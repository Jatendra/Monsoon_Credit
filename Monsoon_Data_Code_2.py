# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 22:53:53 2017

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,VotingClassifier,ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.preprocessing import RobustScaler

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.feature_selection import chi2
#################################################
data=pd.read_csv("E:/Monsoon_Data/X_train.csv")
test=pd.read_csv("E:/Monsoon_Data/X_validation.csv")
ydf=pd.read_csv("E:/Monsoon_Data/y_train.csv")


ydf.shape
data.shape
data.info()

y=ydf[['Dependent_Variable']]
ydf.Dependent_Variable.value_counts()

del ydf['Unique_ID']
Id=test['Unique_ID']

full=pd.concat([data,test])
full.shape
full.index=range(44067)
####################################################
del full['Unique_ID']
full=full.drop(['N32','N26','N27','N25','N31','N30','N29','N28'],axis=1)

full.head()

#..Looking for null values

def null_predict(data):
    total=data.isnull().sum().sort_values(ascending=False)    
    percentage=((data.isnull().sum()*100)/data.isnull().count()).sort_values(ascending=False)
    missing=pd.concat([total,percentage],axis=1,keys=['Total','Percent'])
    return missing

    
null_predict(data)


## N1,N34,N24,N23,N9,N11,N2,N33,N35,N19,N17,N7,C3,N18
## N13,N6,N20,C4,N8,N3,C2,N4,N10,N5,N14

#################################################################

## Categorical variables

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


## Continuous variables

sns.distplot(temp['N23'])

facet = sns.FacetGrid(data=temp, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N1',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp['N1'].max()))
#_________________________________________________________________________
    
## 12,2,16,4,5,18,17,22,21,20,19,23,11,14,7,10,35,13,15,6,3

temp=pd.concat([data,y],axis=1)

sns.distplot(temp['N3'])
sns.distplot(np.square(temp['N3']))

temp.N23.value_counts()

temp['N23']=np.cbrt(temp['N23'])


facet = sns.FacetGrid(data=temp, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N16',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp['N16'].max()))

fig, axis1 = plt.subplots(1,1,figsize=(10,4))
average= temp[["N22", "Dependent_Variable"]].groupby(['N22'],as_index=False).mean()
sns.barplot(x='N22', y='Dependent_Variable', data=average)

#______________________________________________________________________

sns.distplot(temp['N35'])

facet = sns.FacetGrid(data=temp, hue="Dependent_Variable",aspect=4)
facet.map(sns.kdeplot,'N35',shade= True)
facet.add_legend()
facet.set(xlim=(0, temp['N35'].max()))

fig, axis1 = plt.subplots(1,1,figsize=(10,4))
average= temp[["N35", "Dependent_Variable"]].groupby(['N35'],as_index=False).mean()
sns.barplot(x='N35', y='Dependent_Variable', data=average)


## set gap(N1)-->(0-18)(18-38)
## N8-->(0-4)(4-rest)
# Get cube rooted N2,N15,N23
# N3 divide half for 3.25,N6 half by 3.4,N11 half by 12.5,N21 half by .9
# N16-->(0,1)
# N23 half by 3.8

#################################################################

#  Imputation

# N2-->60,120   .0057   .0082
# N4-->10,25    .033    .042
# N5-->11,23    .034    .036
# N7-->36,70    .015
# N8-->?,11     .074
# N9-->?,600    .006
# N10-->1.5,3   .18
# N11-->5,10    .19
# N12-->463,original mean  .038
# N13-->2.5,5   .2
# N14-->2,4     .094
# N15-->1,2     .099
# N16 delete 
# N17-->?,16000 .00048
# N19-->5913,15000 .11
# N20-->20,40    .032
# N22-->4,8      .11
# N23-->26.5,80  .049
# N24-->4197.22,10000  .066
# N34-->?,1400   .034

# N18-->change 4.73,5.83 with mean
# N11,N14,N7,N10,N35,N13,N15,N6,N3
# N12,N2,N16,N4,N5,N18,N17,N22,N21,N20,N19,N23 
# No Chnge-->N1,N3,N21

data['N20'].describe()
data['N20'].value_counts()
a1=data.loc[data['N12']>120,'N2']
b1=a1.value_counts()
b1
sum(b1.index.sort_values())/383


##########################################################################

full.loc[full.N2.isnull(),'N2']=full['N2'].mean()
full.loc[full.N2>120,'N2']=60

full.loc[full.N3.isnull(),'N3']=3.4

full.loc[full.N4.isnull(),'N4']=full.loc[((full['C1']==1)&(full['C2']==0)),'N4'].mean()
full.loc[full.N4>25,'N4']=13

full.loc[(full.N5.isnull())|(full.N5>23),'N5']=12   # .031-->.038

full.loc[full.N6.isnull(),'N6']=full['N6'].mean()

full.loc[(full.N7.isnull())|(full.N7>70),'N7']=36

full.loc[full.N8>11,'N8']=5   # .074-->.11

full.loc[full.N9>600,'N9']=299  #.0068-->.074

full.loc[full.N10.isnull(),'N10']=2
full.loc[full.N10>3,'N10']=4

full.loc[full.N11.isnull(),'N11']=5
full.loc[full.N11>10,'N11']=11

full.loc[full.N13.isnull(),'N13']=3
full.loc[full.N13>5,'N13']=6

full.loc[full.N14.isnull(),'N14']=2
full.loc[full.N14>4,'N14']=5

full.loc[full.N15.isnull(),'N15']=1
full.loc[full.N15>2,'N15']=3

full.loc[(full.N17.isnull())|(full.N17>16000),'N17']=6661.62

full.loc[(full['N18']==4.73)|(full['N18']==5.83)|(full['N18'].isnull()),'N18']=.96

full.loc[full['N19']>15000,'N19']=16000
full.loc[full['N19'].isnull(),'N19']=5913  #.03

full.loc[full['N20']>40,'N20']=20
full.loc[full['N20'].isnull(),'N20']=20  #.03

full.loc[full.N21.isnull(),'N21']=full['N21'].mean()

full.loc[(full.N22>8)|(full.N22.isnull()),'N22']=4

full.loc[(full.N23>80)|(full.N23.isnull()),'N23']=100

full.loc[full.N24>10000,'N24']=full.N24.mean()

full.loc[full.N34>1400,'N34']=full.N34.mean()

full.loc[full.N35.isnull(),'N35']=full['N35'].mean()

full=full.drop(['N12','N16'],axis=1)

##################################################################

#   0.75400803865781507             0.74456770881085543/0.74401760337398004

full.loc[full.N2.isnull(),'N2']=full['N2'].median()
full.loc[full.N2>120,'N2']=60

full.loc[full.N3.isnull(),'N3']=full['N3'].median()

full.loc[full.N4.isnull(),'N4']=full.loc[((full['C1']==1)&(full['C2']==0)),'N4'].median()
full.loc[full.N4>25,'N4']=13

full.loc[(full.N5.isnull())|(full.N5>23),'N5']=full['N5'].median()   # .031-->.038

full.loc[full.N6.isnull(),'N6']=full['N6'].median()

full.loc[(full.N7.isnull())|(full.N7>70),'N7']=36

full.loc[full.N8>11,'N8']=5   # .074-->.11

full.loc[full.N9>600,'N9']=299  #.0068-->.074

full.loc[full.N10.isnull(),'N10']=full.N10.median()
full.loc[full.N10>3,'N10']=4

full.loc[full.N11.isnull(),'N11']=full.N11.median()
full.loc[full.N11>10,'N11']=11

full.loc[full.N13.isnull(),'N13']=full.N13.median()
full.loc[full.N13>5,'N13']=6

full.loc[full.N14.isnull(),'N14']=full.N14.median()
full.loc[full.N14>4,'N14']=5

full.loc[full.N15.isnull(),'N15']=full.N15.median()
full.loc[full.N15>2,'N15']=3

full.loc[(full.N17.isnull())|(full.N17>16000),'N17']=full.N17.median()

full.loc[(full['N18']==4.73)|(full['N18']==5.83)|(full['N18'].isnull()),'N18']=full.N18.median()

full.loc[full['N19']>15000,'N19']=16000
full.loc[full['N19'].isnull(),'N19']=full.N19.median()  #.03

full.loc[full['N20']>40,'N20']=20
full.loc[full['N20'].isnull(),'N20']=full.N20.median()  #.03

full.loc[full.N21.isnull(),'N21']=full['N21'].median()

full.loc[(full.N22>8)|(full.N22.isnull()),'N22']=full['N22'].median()

full.loc[(full.N23>80)|(full.N23.isnull()),'N23']=full['N23'].median()

full.loc[full.N24>10000,'N24']=full.N24.median()

full.loc[full.N34>1400,'N34']=full.N34.median()

full.loc[full.N35.isnull(),'N35']=full['N35'].median()


full=full.drop(['N12','N16'],axis=1)

##################################################################

data=full.loc[0:33050]
temp=pd.concat([data['N16'],y],axis=1)

corra=temp.corr()
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(corra,annot=True)

###################################################################

data=full.loc[0:33050]

data.N23.value


# 22,34,46,66,
####################################################################

# Imputation by KNN

data=data.dropna()

temp=pd.concat([data,y],axis=1)

corra=temp.corr()
fig, ax = plt.subplots(figsize=(30,24))
sns.heatmap(corra,annot=True)

# less  N11,N14,N10,N35,N13,N15,N6
# More  N12,N2,N16,N4,N17,N22,N21,N20,N19,N23
# No    N9,N8,N1,N24,N33,N34
# New   N23,N13,N4,N22,N11,

# N2-->N35l,N20m>>
# N4-->N8n,N20m,N9n>>
# N6-->N1n,N19m,N21m >>
# N10-->N21m,N14l,N12m>>
# N11-->N13l,N6l(may be)>>
# N13-->N11l,N22m>>
# N14-->N21m,N10l>>
# N15-->N21m,N6l>>
# N17-->N9n,N4m,N24n,N8n>>
# N19-->N6l,N1n >>
# N20-->N4m,N8n>>
# N21-->N14l,N10l,N6l>>
# N22-->N13l,N4m>>
# N23-->N24>>
# N35-->N20m,N2m>>

## 0.75721446958406724              0.74456770881085543

data=full[0:33050]

scaler=RobustScaler()

train=scaler.fit_transform(data.loc[data.N23.notnull(),['N24']])
test=full.loc[full.N23.isnull(),['N24']]
ya=data.loc[data.N23.notnull(),['N23']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N23.isnull(),['N23']]=pred

#__________________________________________________________________

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

#_________________________________________________________________

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N11.notnull(),['N13']])
test=full.loc[full.N11.isnull(),['N13']]
ya=data.loc[data.N11.notnull(),['N11']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N11.isnull(),['N11']]=pred

#__________________________________________________________________

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

#________________________________________________________________

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N15.notnull(),['N6','N21']])
test=full.loc[full.N15.isnull(),['N6','N21']]
ya=data.loc[data.N15.notnull(),['N15']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N15.isnull(),['N15']]=pred

#________________________________________________________________

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N19.notnull(),['N6','N1']])
test=full.loc[full.N19.isnull(),['N6','N1']]
ya=data.loc[data.N19.notnull(),['N19']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N19.isnull(),['N19']]=pred

#________________________________________________________________

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N20.notnull(),['N4','N8']])
test=full.loc[full.N20.isnull(),['N4','N8']]
ya=data.loc[data.N20.notnull(),['N20']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N20.isnull(),['N20']]=pred

#________________________________________________________________

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N17.notnull(),['N4','N8','N9','N24']])
test=full.loc[full.N17.isnull(),['N4','N8','N9','N24']]
ya=data.loc[data.N17.notnull(),['N17']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N17.isnull(),['N17']]=pred

#________________________________________________________________

full.loc[full.N35.isnull(),'N35']=full['N35'].median()

data=full[0:33050]

train=scaler.fit_transform(data.loc[data.N2.notnull(),['N35','N20']])
test=full.loc[full.N2.isnull(),['N35','N20']]
ya=data.loc[data.N2.notnull(),['N2']]
model=KNR()
model.fit(train,ya)
pred=model.predict(test)

full.loc[full.N2.isnull(),['N2']]=pred

#________________________________________________________________

full=full.drop(['N12','N18','N16','N5','N7','N3'],axis=1)
#_________________________________________________________________
data=full[0:33050]

temp=pd.concat([data,y],axis=1)

corra=temp.corr()
fig, ax = plt.subplots(figsize=(28,16))
sns.heatmap(corra,annot=True)

# .11

###################################################################
full.loc[:,'N16X']=0
full.loc[full['N16']!=0,'N16X']=1

'''

full.loc[:,'N1X']=0
full.loc[(full['N1']>18),'N1X']=1

full.loc[:,'N8X']=0
full.loc[(full['N1']>4),'N8X']=1

full.loc[:,'C6C']=0
full.loc[full['C6']==True,'C6X']=1

full.loc[:,'C8C']=0
full.loc[full['C8']==True,'C8X']=1

full.loc[:,'N3X']=0
full.loc[full['N3']>3.25,'N3X']=1

full.loc[:,'N6X']=0
full.loc[full['N6']>3.4,'N6X']=1

full.loc[:,'N11X']=0
full.loc[full['N11']>12.5,'N11X']=1

full.loc[:,'N21X']=0
full.loc[full['N21']>.9,'N21X']=1

full.loc[:,'N23X']=0
full.loc[full['N23']>3.8,'N23X']=1

'''

####################################################################

temp1=temp.loc[temp.N23.notnull()]
temp1.index=range(30564)

a1=temp1.loc[0:3056,'N23'].mean()      #1322
b1=temp1.loc[0:3056,'Dependent_Variable'].mean()  #.3016
a2=temp1.loc[3057:6112,'N23'].mean()   #979
b2=temp1.loc[3057:6112,'Dependent_Variable'].mean() #.2891
a3=temp1.loc[6113:9168,'N23'].mean()   #923
b3=temp1.loc[6113:9168,'Dependent_Variable'].mean() #.2922
a4=temp1.loc[9169:12224,'N23'].mean()  #955
b4=temp1.loc[9169:12224,'Dependent_Variable'].mean() #.2912
a5=temp1.loc[12225:15280,'N23'].mean() #1077
b5=temp1.loc[12225:15280,'Dependent_Variable'].mean() #.2989
a6=temp1.loc[15281:18336,'N23'].mean() #885
b6=temp1.loc[15281:18336,'Dependent_Variable'].mean() #.2975
a7=temp1.loc[18337:21392,'N23'].mean() #841
b7=temp1.loc[18337:21392,'Dependent_Variable'].mean() #.3013
a8=temp1.loc[21393:24448,'N23'].mean() #1123
b8=temp1.loc[21393:24448,'Dependent_Variable'].mean() #.2954
a9=temp1.loc[24449:27504,'N23'].mean() #1035
b9=temp1.loc[24449:27504,'Dependent_Variable'].mean() #.2831
a10=temp1.loc[27505:30564,'N23'].mean() #1015
b10=temp1.loc[25588:30564,'Dependent_Variable'].mean() #.2931

x=[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10]
y=[b1,b2,b3,b4,b5,b6,b7,b8,b9,b10]

plt.plot(x,y)

33050-2486
##################################################################

# data2=data[['N1','N34','N24','N23','N9','N11','N2','N33','N35','N19','N17','N7','C3','N18']]

data=full[0:33050]

train_x,test_x,train_y,test_y=train_test_split(data,y,test_size=0.33,random_state=25)

model=RandomForestClassifier(random_state=42)
model.fit(train_x,train_y)
model.score(train_x,train_y) #98
model.score(test_x,test_y)   # 71

#  N1--(72,69) ,
#  add N34 (83,66) N24 (89,65)  N23 (83,65) N9 (91,66) N11 (80,66) N2 (85,65) N33 (82,65)
#  N35 (81,65) N19 (91,66)  N17 (91,66)  N7(84,65)  C3(81,66)  N18(85,65)
  

model1=GradientBoostingClassifier(random_state=42)
model1.fit(train_x,train_y)
model1.score(train_x,train_y)  #75
model1.score(test_x,test_y)    #74


model2=LogisticRegression(random_state=42)
model2.fit(train_x,train_y)
model2.score(train_x,train_y)   #underfitting 72
model2.score(test_x,test_y)

data.columns

##_____________________________________________________________________


## Parameter tuning 
 
rf_param_grid = {"n_estimators" :[350],
              "max_depth": [7],
              "max_features": [None],
              "min_samples_split" :[2]
                }

gsRFC = GridSearchCV(model,param_grid = rf_param_grid, cv=10, scoring="accuracy", n_jobs= 1, verbose = 1)

gsRFC.fit(train_x,train_y)

RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_  #72.9 at depth=7

RFC_best.score(train_x,train_y)  #74.7
RFC_best.score(test_x,test_y)  #73.52

gsRFC.grid_scores_


########################################################################

#  Feature selection

'''
C1 .063
C2 .052
C3 .0099
C4 .054
C5 .0014
C6 .032
C7 .0067
C8 .029
N1 .27     >>>
N2 .0014   <<<
N4 .032    <<<
N6 .26     >>>
N8 .074    <<<
N9 .0068   <<<
N10 .18    >>>
N11 .18    >>>
N13 .2     >>>
N14 .093   <<<
N15 .098   >>>
N17 .014   <<<
N19 .12    >>>
N20 .07    <<<
N21 .16    <<<
N22 .14    <<<
N23 .045   >>>
N24 .066   >>>
N33 .014   <<<
N34 .034   >>>
N35 .01    <<<
'''

## (75.36,74.26)--(75.41,74.38)
data=full[0:33050]

# Create ANOVA F-Values
f_classif(data, y)

data.columns


##  N1,N6,N8,N10,N11,N13,N14,N15,N19,N20,N21,N22,C2,C3,C4

data2=data[['C2','C3','C4','C5','C6','C7']]

data3 = data2.astype(int)

chi2(data3,y)


##########################################################################
# So final data is

data1=data[['C4','C2','C3','C6','N1','N6','N13','N11','N10','N22','N15','N24','N23','N34','N19']]
data1=data.drop(['N2'],axis=1)
y=np.ravel(y)

train_x,test_x,train_y,test_y=train_test_split(data1,y,test_size=0.2,random_state=25)

model=GradientBoostingClassifier(random_state=42)
model.fit(train_x,train_y)
model.score(train_x,train_y)  #75
model.score(test_x,test_y)    #74


# Gradient boosting tunning


gb_param_grid = {'n_estimators' : [200],
                 'learning_rate': [0.1],
                 'max_depth': [3]
                }

gsGBC = GridSearchCV(model,param_grid = gb_param_grid, cv=10, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(train_x,train_y)

GBC_best = gsGBC.best_estimator_  #(300)(.05)(8)(150)(0.3)

# Best score
gsGBC.best_score_         #(0.73716)

GBC_best.score(train_x,train_y)
GBC_best.score(test_x,test_y)  #(0.735766)

gsGBC.grid_scores_

'''
'learning_rate': [0.1, 0.05, 0.01],
                 '',
                'min_samples_leaf': [100,150],
                'max_features': [0.3, 0.1]
'''

####################################################################

#Feature importance


importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,8))
plt.title("Feature importances")
plt.bar(range(train_x.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(train_x.shape[1]), indices)
plt.xlim([-1, train_x.shape[1]])
plt.show()

########################################################################

kfold = StratifiedKFold(n_splits=10)

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


g = plot_learning_curve(RFC_best,"RF learning curves",train_x,train_y,cv=kfold)
g = plot_learning_curve(model1,"GBM learning curves",train_x,train_y,cv=kfold)


######################################################################

test=full.loc[33050:]
result=model1.predict_proba(test)
output=pd.DataFrame([result[i][1] for i in range(11017)],columns=['Class_1_Probability'])
output[:5]
out=pd.concat([Id,output],axis=1)
out.to_csv("E:/Monsoon_Data/Result.csv",index=False)

