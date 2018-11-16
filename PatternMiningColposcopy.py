# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:11:33 2018

@author: joao-
"""

from dummy_var import preprocessData as ppd
import pandas as pd, numpy as np
from IPython.display import display, HTML
from sklearn.preprocessing import LabelBinarizer
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

DATA_PATH = 'C:\\Users\\joao-\\OneDrive\\Documentos\\Ciencia de Dados\\Project\\CD\\datasets\\col\\'
#doc1=pd.read(DATA_PATH+"green.csv", )
#doc2
#doc3
df_green = pd.read_csv(DATA_PATH+'green.csv').drop(columns=['experts::0', 'experts::1', 'experts::2', 'experts::3', 'experts::4', 'experts::5'])
df_hinselmann = pd.read_csv(DATA_PATH+'hinselmann.csv').drop(columns=['experts::0', 'experts::1', 'experts::2', 'experts::3', 'experts::4', 'experts::5'])
df_schiller = pd.read_csv(DATA_PATH+'schiller.csv').drop(columns=['experts::0', 'experts::1', 'experts::2', 'experts::3', 'experts::4', 'experts::5'])
df = pd.concat([df_green, df_hinselmann, df_schiller]).reset_index(drop=True)
#df = ppd(df)
#print(X)
#print(df.columns)

y = df.iloc[:,len(df.columns)-1].values#(columns=['consensus'])
X = df.iloc[:,0:len(df.columns)-1].values
#print(y)
X = MinMaxScaler().fit_transform(X)
X = pd.DataFrame (X)
#

selector= SelectKBest(score_func=chi2,k=4)
X_new = selector.fit_transform(X,y)
idxs_selected=selector.get_support(indices=True)
columns = []



for idx in idxs_selected:
    columns.append(df.columns[idx])
df = pd.DataFrame(data=X_new, columns=columns)
#X_new = ppd(X_new)

for col in list(df) :
    df[col] = pd.qcut(df[col],2,labels=False, duplicates='drop')
    #False, duplicates='drop')#,'4'])#,'5','6'])#,'7','8','9'])
    attrs = []
    values = df[col].unique().tolist()
    values.sort()
    for val in values : 
        attrs.append("%s:%s"%(col,val))
    lb = LabelBinarizer().fit_transform(df[col])
    if(len(attrs)==2) :
        v = list(map(lambda x: 1 - x, lb))
        lb = np.concatenate((lb,v),1)
    df2 = pd.DataFrame(data=lb, columns=attrs)
    df = df.drop(columns=[col])
    df = pd.concat([df,df2], axis=1, join='inner')
with pd.option_context('display.max_rows', 10, 'display.max_columns', 8): print(df)

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True, )
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
display(HTML(rules.to_html()))
with pd.option_context('display.max_rows', 10, 'display.max_columns', 10): 
    (rules.sort_values(by=['support','lift','leverage'],ascending =False).to_csv(path_or_buf='Colposcopy\\colposcopy_rules2.csv'))

r = (rules.iloc[:,[4,6]])

lifts_2=[]
lifts_3=[]
lifts_4=[]
lifts_5=[]
lifts_6=[]
lifts_7=[]
lifts_8=[]
print (r)


for index, row in r.iterrows():
    
    if row['support']>0.2:
        lifts_2.append(row['lift'])
    if row['support']>0.3:
        lifts_3.append(row['lift'])
    if row['support']>0.4:
        lifts_4.append(row['lift'])
    if row['support']>0.5:
        lifts_5.append(row['lift'])
    if row['support']>0.6:
        lifts_6.append(row['lift'])
    if row['support']>0.7:
        lifts_7.append(row['lift'])
    if row['support']>0.8:
        lifts_8.append(row['lift'])

print (sum(lifts_2)/len(lifts_2))
print (sum(lifts_3)/len(lifts_3))
print (sum(lifts_4)/len(lifts_4))
#print (sum(lifts_5)/len(lifts_5))
#print (sum(lifts_6)/len(lifts_6))
#print (sum(lifts_7)/len(lifts_7))
if len(lifts_8) != 0:
    print (sum(lifts_8)/len(lifts_8))
    
print()

print(len(lifts_2))
print(len(lifts_3))
print(len(lifts_4))
print(len(lifts_5))
print(len(lifts_6))
print(len(lifts_7))
print(len(lifts_8))

print(rules[1:50])

line1, = plt.plot(rules[1:50]['support'],rules[1:50]['lift'] , 'b', label='Train AUC')
plt.xlabel('Minimum support')
plt.savefig('a.png', dpi=100)
plt.show()

#    
#    
#for col in list(df) :
#    if col not in ['class'] :
#        df[col] = pd.cut(df[col],3,labels=['0','1','2'])
#    attrs = []
#    values = df[col].unique().tolist()
#    values.sort()
#    for val in values : 
#        attrs.append("%s:%s"%(col,val))
#    lb = LabelBinarizer().fit_transform(df[col])
#    if(len(attrs)==2) :
#        v = list(map(lambda x: 1 - x, lb))
#        lb = np.concatenate((lb,v),1)
#    df2 = pd.DataFrame(data=lb, columns=attrs)
#    df = df.drop(columns=[col])
#    df = pd.concat([df,df2], axis=1, join='inner')
#with pd.option_context('display.max_rows', 10, 'display.max_columns', 8): print(df)
#
#frequent_itemsets = apriori(df, min_support=0.7, use_colnames=True)
##print(frequent_itemsets)
#rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
#display(HTML(rules.to_html()))