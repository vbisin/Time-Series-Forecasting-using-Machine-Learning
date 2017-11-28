#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 19:50:56 2016

@author: albertobisin
"""

import pickle
import numpy as np
#AMEXdict=pickle.load(open("AMEXdict.p","rb"))
#AMEXquotes=pickle.load(open("AMEXquotes.p","rb"))
#NYSEdict=pickle.load(open("NYSEdict.p","rb"))
#NYSEquotes=pickle.load(open("NYSEquotes.p","rb"))
NASDAQdict=pickle.load(open("NASDAQdict.p","rb"))
NASDAQquotes=pickle.load(open("NASDAQquotes.p","rb"))
Stockdict= NASDAQdict.copy()
Stockquotes=list(NASDAQquotes)


N=len(Stockdict)
subdict={}
subquotes=list()
key  = Stockdict.keys()

# delete fund or funds from name or industry 
for i in range(N):
    if Stockdict[key[i]][1][1] != "Closed End Funds":
        subdict[key[i]]=Stockdict[key[i]]
        subquotes.append(Stockquotes[i])
        
Stockdict=subdict
Stockquotes=subquotes
        
pickle.dump(Stockdict, open("NASDAQdictNoFund.p", "wb"))
pickle.dump(Stockquotes, open("NASDAQquotesNoFund.p", "wb"))
  
## Part II run regression on the features 


#Import relevant stock market
AMEXdict=pickle.load(open("AMEXdictNoFund.p","rb"))
AMEXquotes=pickle.load(open("AMEXquotesNoFund.p","rb"))
#NASDAQdict=pickle.load(open("NASDAQdictNoFund.p","rb"))
#NASDAQquotes=pickle.load(open("NASDAQquotesNoFund.p","rb"))
#NYSEdict=pickle.load(open("NYSEdictNoFund.p","rb"))
#NYSEquotes=pickle.load(open("NYSEquotesNoFund.p","rb"))


# Get y vector 
coefs=list()
y=np.zeros(1495)
from sklearn import linear_model
from SVM_Feature_two import Feature_Calling

counter=0
for quote in AMEXquotes:
    error=0
    for i in range (1495):
        y[i]=quote.close[i]-quote.open[i]
    
    k,d,slow_d,momentum,roc,williams_r,ad_oscillator,disparity5,disparity10,oscp,cci,rsi = Feature_Calling(quote.high,quote.close,quote.low,quote.open)
    
    X = np.vstack((k,d))
    X = np.vstack((X,slow_d))
    X = np.vstack((X,momentum))
    X = np.vstack((X,roc))
    X = np.vstack((X,williams_r))
    X = np.vstack((X,ad_oscillator))
    X = np.vstack((X,disparity5))
    X = np.vstack((X,disparity10))
    X = np.vstack((X,oscp))
    X = np.vstack((X,cci))
    X = np.vstack((X,rsi))
    X = np.transpose(X)
    
    reg=linear_model.LinearRegression(normalize=True)
    #reg=linear_model.Ridge(alpha=0.5)
    #reg=linear_model.Lasso(alpha=.1)
    try: 
        reg.fit(X,y)
        coefs.append(reg.coef_)
    except:
        counter+=1
        
coefs=np.asarray(coefs)

for j in range(len(X[1,:])):
#ie variable   
    sum=0
    for i in range(len(coefs)):
    # ie sample    
        sum=sum+coefs[i,j]
    print("average for variable "+str(j+1)+" is: "+str(1000*sum/len(coefs[:,1])))
print(counter)




