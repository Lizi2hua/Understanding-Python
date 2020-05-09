# -*- coding: utf-8 -*-
"""
Created on Sat May  9 17:15:30 2020

@author: Natuski_
"""
#*******************************#
import pandas as pd

#*************read data**********#
data=pd.read_csv('C:\\Users\\Administrator\\Desktop\\Project：777\\data\\pokemon-challenge\\pokemon.csv')
datahead=data.head() #show first 5 rows
print(datahead)
datatial=data.tail()
print(datatial)#show last 5 rows
datacolumn=data.columns
print(datacolumn)
#*******************************#
