# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:15:30 2020

@author: Natuski_
"""
#*******************************#
import pandas as pd

#*************read data**********#
data=pd.read_csv('C:\\Users\\Administrator\\Desktop\\Project：777\\data\\pokemon-challenge\\pokemon.csv')
#data=pd.read_csv('C:\\Users\\李梓桦\\Desktop\\培训V20200507\\dataset\\pokemon-challenge\\pokemon.csv')
datahead=data.head() #show first 5 rows
print(datahead)
datatial=data.tail()
print(datatial)#show last 5 rows
datacolumn=data.columns
print(datacolumn)
print(data.shape)
print("-------------------")
print(data.info)
#look frequency of pokemon types
print(data['Type 1'].value_counts(dropna=False))
#value_count（）需要指定对哪一列使用，这里是data中的Type 1
#*******************************#
data_describe=data.describe()
data.boxplot(column='Attack',by='Legendary')
#分离指定的列 data[['col1','col2']]
dat2=data[['Attack','Legendary']]