# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 19:38:44 2016

@author: raghavsinghal
"""

import numpy as np
from yahoo_finance import Share
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import pickle


AMEXdict = pickle.load(open("amexCluster.p","rb"))
NYSEdict = pickle.load(open("nyseCluster.p","rb"))
NASDAQdict = pickle.load(open("nasdaqCluster.p","rb"))

""""
Momentum over a 15 day period 
"""
def K(H,C,L): 
    N=15

    n = len(H)
    k = np.zeros(n-N)
    
    for i in range(n-N):
        k[i] = 100*(C[i+N] - np.min(L[i:i+1+N]))/(np.max(H[i:i+1+N]) - np.min(L[i:i+1+N]))
    return k 

def D(k,H):
    N=15

    n = len(k) - N
    d = np.zeros(n)
    
    for i in range(n):
        d[i] = (1./N)*np.sum(k[N + 1 + i:i + 2*N + 1])
    
    d = np.hstack((np.ones(N)*d[0] , d))

    return d
        
def Slow_D(d):
    N=15

    n = len(d) - N 
    slow_d = np.zeros(n)
    
    for i in range(n):
        slow_d[i] = (1./N)*np.sum(d[N + 1 + i:i + 2*N + 1])
        
    slow_d = np.hstack((np.ones(N)*slow_d[0] , slow_d))
    return slow_d
    
def Momentum(C):
    N = 15
    
    momentum = C[4:] - C[:-4]
    return momentum[N-4:]
    
def ROC(C):
    N = 15
    
    roc = (100.)*(C[N:]/C[:-N])
    
    return roc
    
def Williams_R(H,C,L):
    N = 15
    n = len(C)
    
    williams_r = np.zeros(n)


    for i in range(n-N):
        a = np.max(H[i:i + N]) - np.min(L[i:i + N])
        if a==0:
            williams_r[i] = williams_r[i-1]
        else:    
            williams_r[i] = 100*(np.max(H[i:i + N]) - C[i])/(a)
        
    for i in range(n-N,n):
        a = (np.max(H[i - N :i]) - np.min(L[i - N:i]))
        
        if a == 0:
            williams_r[i] = williams_r[i-1]
        else:    
            williams_r[i] = 100*(np.max(H[i - N:i]) - C[i])/a
    return williams_r[N:]
    
def AD_oscillator(H,C,L):
    n = len(C)
    N=15
    
    ad_oscillator = np.zeros(n)
    
    if (H[0] - L[0])==0:
        ad_oscillator[0]=0
    else:
        ad_oscillator[0] = (H[0]-C[0])/(H[0] - L[0])
    
    
    for i in range(1,n):
        denom = H[i] - L[i]
        num = H[i] - C[i-1]
        if denom == 0:
            ad_oscillator[i] = ad_oscillator[i-1]
        else:   
            ad_oscillator[i] = num/denom
    #ad_oscillator[1:] = (H[1:] - C[:-1])/(H[1:] - L[1:])
    return ad_oscillator[N:]
    
def Disparity5(C):
    n = len(C)
    N = 15
    
    MA = np.zeros(n)
    
    for i in range(5):
        MA[i] = 0.2*(np.sum(C[i:i+5]))
    
    for i in range(5,n):
        MA[i] = 0.2*(np.sum(C[i-5:i]))

    
    disparity5 = 100*C/(MA)
    return disparity5[N:]

def Disparity10(C):
    n = len(C)    
    N =15
    
    MA = np.zeros(n)
    
    for i in range(5):
        MA[i] = 0.2*(np.sum(C[i:i+5]))
    
    for i in range(5,n):
        MA[i] = 0.2*(np.sum(C[i-5:i]))

    disparity10 = 100*C/(MA)
    
    return disparity10[N:]

def OSCP(C):
    n = len(C)
    N = 15 

    MA5 = np.zeros(n)
    
    for i in range(5):
        MA5[i] = 0.2*(np.sum(C[i:i+5]))
    
    for i in range(5,n):
        MA5[i] = 0.2*(np.sum(C[i-5:i]))

    
    MA10 = np.zeros(n)
    
    for i in range(10):
        MA10[i] = 0.1*(np.sum(C[i:i+10]))
    
    for i in range(10,n):
        MA10[i] = 0.2*(np.sum(C[i-10:i]))

    oscp = (MA5 - MA10)/MA5
    
    return oscp[N:]
    
def CCI(H,C,L):
    M = (H+C+L)/3.
    
    N = 15
    n = len(C)
    
    SM = np.zeros(n - N)
    D = np.zeros(n-N)    
    
    for i in range(n - N):
        SM[i] = (1./N)*np.sum(M[i:i + N + 1])
        
        l = len(M[i:i + N + 1])
        a = np.ones(l)*SM[i]
        
        D[i] = (1./N)*np.sum(np.abs(M[i:i + N + 1] - a))
        
    cci = (M[N:] - SM)/(0.015*D)
    
    return cci

def RSI(C,O):
    N = 15
    n = len(C)
    
    Up = np.zeros(n-N)
    Dw = np.zeros(n-N)
    
    for i in range(n-N):
        for j in range(N):
            if C[i + j] - O[i+j] >= 0:
                Up[i] += C[i + j] - O[i+j]
            elif C[i + j] - O[i+j] < 0:
                 Dw[i] +=   -(C[i + j] - O[i+j])
                 
    Up = (1./15)*Up
    Dw = (1./15)*Dw
    
    Rs = Up/Dw
    
    rsi = 100.0 - (100./(1 + Rs))
    
    return rsi
    
    

def Feature_Calling(H,C,L,O):
    k = K(H,C,L)
    d = D(k,H)
    slow_d = Slow_D(d)
    momentum = Momentum(C)
    roc = ROC(C)
    williams_r = Williams_R(H,C,L)
    ad_oscillator = AD_oscillator(H,C,L)
    disparity5 = Disparity5(C)
    disparity10 = Disparity10(C)
    oscp = OSCP(C)
    cci = CCI(H,C,L)
    rsi = RSI(C,O)
    
    
    
    return k,d,slow_d,momentum,roc,williams_r,ad_oscillator,disparity5,disparity10,oscp,cci,rsi

def IO(H,C,L,O):
        
    Y = np.sign(C[1:] - C[:-1])
    
    k,d,slow_d,momentum,roc,williams_r,ad_oscillator,disparity5,disparity10,oscp,cci,rsi = Feature_Calling(H,C,L,O)
    
    
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
        
    return X,Y
    
def cCompany(comp,nStockdict):
    Cnumber = nStockdict[comp][2]
    listComp = list()
    
    dash = nStockdict.keys()
    n = len(dash)
    for i in range(n):
        if nStockdict[dash[i]][2] == Cnumber:
            listComp.append(dash[i])
    return listComp

def clusterPrice(comp, start, end,nStockdict):
    listComp = cCompany(comp,nStockdict)
    
    H,C,L,O,V = obtainHCLO(listComp[0],start,end)

    for c in listComp:
        if c!=listComp[0]:
            high,low,openprice,close,volume = obtainHCLO(c,start,end)
            H+= high  
            C+=low  
            L+= openprice  
            O += close
            V += volume
            
    return H,C,L,O
        


def obtainHCLO(comp, start, end):
    stock = Share(comp)
    
    history = stock.get_historical(start, end)
    
    n = len(history)
    
    H,C,L,O,V = np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)
    
    for i in range(n):
        H[i] = history[-(i+1)]['High']
        C[i] = history[-(i+1)]['Close']
        L[i] = history[-(i+1)]['Low']
        O[i] = history[-(i+1)]['Open']
        V[i] = history[-(i+1)]['Volume']
    return H,C,L,O,V


def stockdata(comp,percentage, start, end,marketDict):
    stock = Share(comp)
    
    history = stock.get_historical(start, end)
    
    n = len(history)
    
    H,C,L,O = np.zeros(n),np.zeros(n),np.zeros(n),np.zeros(n)
    
    for i in range(n):
        H[i] = history[-(i+1)]['High']
        C[i] = history[-(i+1)]['Close']
        L[i] = history[-(i+1)]['Low']
        O[i] = history[-(i+1)]['Open']
    
    H,C,L,O,V = obtainHCLO(comp,start,end)

    Y5 = np.sign(C[5:] - C[:-5])
    Y10 = np.sign(C[10:] - C[:-10])

        
    X,Y = IO(H,C,L,O)
    H,C,L,O = clusterPrice(comp,start,end,marketDict)
    
    Xcluster , Ycluster = IO(H,C,L,O)        
    
    X = np.hstack((X,Xcluster))
    
    N = 15 
    X1 = X[:-1,:]
    X5 = X[:-5,:]
    X10 = X[:-10,:]    
    
    Y1 = Y[N:]
    Y5 = Y5[N:]
    Y10 = Y10[N:]
    

        
    sample = int(len(X)*percentage/100.)
    
    Xtrain1,Ytrain1 = X1[:sample,:],Y1[:sample]
    Xtrain5,Ytrain5 = X5[:sample,:],Y5[:sample]
    Xtrain10,Ytrain10 = X10[:sample,:],Y10[:sample]

    
    Xtest1,Ytest1 = X1[sample:,:],Y1[sample:]
    Xtest5,Ytest5 = X5[sample:,:] , Y5[sample:]
    Xtest10,Ytest10 = X10[sample:,:] , Y10[sample:]
    
    
    np.savetxt(comp +'1_xtrain.txt',Xtrain1)
    np.savetxt(comp +'1_ytrain.txt',Ytrain1)
    np.savetxt(comp +'1_xtest.txt',Xtest1)
    np.savetxt(comp +'1_ytest.txt',Ytest1)

    np.savetxt(comp +'5_xtrain.txt',Xtrain5)
    np.savetxt(comp +'5_ytrain.txt',Ytrain5)
    np.savetxt(comp +'5_xtest.txt',Xtest5)
    np.savetxt(comp +'5_ytest.txt',Ytest5)

    np.savetxt(comp +'10_xtrain.txt',Xtrain10)
    np.savetxt(comp +'10_ytrain.txt',Ytrain10)
    np.savetxt(comp +'10_xtest.txt',Xtest10)
    np.savetxt(comp +'10_ytest.txt',Ytest10)

    
    
def SVM_poly(comp,window):
    
    xtrain = np.loadtxt(comp + str(window)  + "_xtrain.txt")
    ytrain = np.loadtxt(comp + str(window)  + "_ytrain.txt")
    xtest = np.loadtxt(comp + str(window)  + "_xtest.txt")
    ytest = np.loadtxt(comp + str(window)  + "_ytest.txt")
    
    min_max_scaler = preprocessing.MinMaxScaler([-1,1])
    
    xtrain = min_max_scaler.fit_transform(xtrain)
    xtest = min_max_scaler.fit_transform(xtest)

    c = 2**np.arange(2,10,dtype =float)
    
    D = np.arange(1,5)
    
    n = len(c)
    m = len(D)
    
    MeanMatrix = np.zeros((n,m))
    Deviation = np.zeros((n,m))
    
    for i in range(n):
        for j in range(m):
            
            clf = SVC(C = c[i], kernel = 'poly', degree = D[j])
            
            score = cross_val_score(clf, xtrain, ytrain, cv = 3)
            score = np.ones(3) - score
            MeanMatrix[i,j] = np.mean(score)
            Deviation[i,j] = np.std(score)
            
    row = 0
    col = 0 
    min = MeanMatrix[row,col]
    
    for i in range(n):
        for j in range(m):
            if MeanMatrix[i,j] <= min:
                row,col = i,j
                min = MeanMatrix[i,j]
    C_star = c[row]
    d_star = D[col]
    

  
    # train error for fixed c_star
    trainc =np.zeros(len(D))
    for i in D:
        clf = SVC( C = C_star, kernel = 'poly', degree = i)
        clf.fit(xtrain,ytrain)

        trainc[i-1] = 1 - clf.score(xtrain,ytrain)
    
     # train error for fixed d_star
    traind =np.zeros(len(c))
    for i in range(len(c)):
        clf = SVC( C = c[i], kernel = 'poly', degree = d_star)
        clf.fit(xtrain,ytrain)

        traind[i-1] = 1 - clf.score(xtrain,ytrain)
        

     
    # plot train and test errors for a fixed cstar
    plt.figure(1)
    plt.plot(D,MeanMatrix[row,:],'r',label='test errors')
    plt.plot(D,trainc,'b',label='train errors')
    plt.legend(bbox_to_anchor=(.65,0.3), loc=2,borderaxespad=0.)
    plt.title("Train and test errors for a fixed c*")
    plt.xlabel('degree values')
    plt.ylabel('Errors')
    plt.savefig(comp + ' '+ str(window) +'-day window error plot 1')
    
        # plot train and test errors for a fixed dstar
    plt.figure(2)
    plt.plot(c,MeanMatrix[:,col],'r',label="test errors")
    plt.plot(c,traind,'b',label="train errors")
    plt.legend(bbox_to_anchor=(.65,.3), loc=2,borderaxespad=0.)
    plt.title("Train and test errors for a fixed D*")
    plt.xlabel('c values')
    plt.ylabel('Errors')
    plt.savefig(comp + ' '+ str(window) +'-day error plot 2')
    plt.show()
   
   
   
   
   # plt.xlabel("Error rate for C =" + str(C_star))

    clf = SVC( C = C_star, kernel = 'poly', degree = d_star)
        
    clf.fit(xtrain,ytrain)
    
    
    print ('Test Error is' ),(1 - clf.score(xtest,ytest))
    print  ("C* is"),C_star
    print ("D* is "),d_star
        

def runmodel(share,marketDict,window):
   #stockdata(share,80,'2014-01-01','2016-12-09',marketDict)
   SVM_poly(share,window)
   

runmodel('FDX',NYSEdict,10)


"""
NYSE
Citigroup, Inc. Common Stock - C
DTE Energy Company Common Stock - DTE
Mueller Industries, Inc. Common - MLI
Exxon mobil -XOM
FedEx - FDX
"""
"""
NASDAQ
SB Financial Group, Inc. -SBFG
Abraxas Petroleum Corporation - AXAS
Microsoft Corporation - MSFT
Titan Pharmaceuticals, Inc. - TTNP
Viacom Inc. - VIA
"""

"""
AMEX
Mexco Energy Corporation Common -MXC
NanoViricides, Inc. NEW Common -NNVC
BioTime, Inc. Common Stock -BTX
U.S. Geothermal Inc. Common Sto - HTM
China Pharma Holdings, Inc. Com - CPHI
"""

