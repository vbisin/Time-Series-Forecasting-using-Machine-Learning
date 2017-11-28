
import pickle
import numpy as np
from sklearn import cluster, covariance
 
    
## Choose stock Market 
AMEXdict=pickle.load(open("AMEXdictNoFund.p","rb"))
AMEXquotes=pickle.load(open("AMEXquotesNoFund.p","rb"))
#NYSEdict=pickle.load(open("NYSEdictNoFund.p","rb"))
#NYSEquotes=pickle.load(open("NYSEquotesNoFund.p","rb"))
#NASDAQdict=pickle.load(open("NASDAQdictNoFund.p","rb"))
#NASDAQquotes=pickle.load(open("NASDAQquotesNoFund.p","rb"))
Stockdict=AMEXdict
Stockquotes=AMEXquotes


##Get array with names of stocks
val = Stockdict.values()
names = list()
for i in range(len(val)):
   names.append(val[i][0])  
names=np.asarray(names)


###############################################################################



##Iterate over feature
from SVM_Feature_two import Momentum,ROC,AD_oscillator,Disparity5,Disparity10,OSCP,RSI
feature=[OSCP]

featureMatrix=np.ones((1485,len(Stockquotes)))
#counter=0

amexF1 = np.zeros((99,len(Stockquotes)))
amexF2 = np.zeros((99,len(Stockquotes)))
amexF3 = np.zeros((99,len(Stockquotes)))
amexF4 = np.zeros((99,len(Stockquotes)))
amexF5 = np.zeros((99,len(Stockquotes)))
amexF6 = np.zeros((99,len(Stockquotes)))
amexF7 = np.zeros((99,len(Stockquotes)))


N = len(feature)

for i in range(N):
    for j  in range(len(Stockquotes)):
        featureMatrix[:,j]=feature[i](Stockquotes[j].high[:1500],Stockquotes[j].low[:1500],Stockquotes[j].close[:1500],Stockquotes[j].open[:1500])
        #counter+=1
        featureMatrix[:,0] = featureMatrix[:,1]  
    #featureMatrix=np.transpose(featureMatrix)    
###########################################
#Iterate over time interval (break into days of 15)
    for k in range(99):
        sampleFeatureMatrix=featureMatrix[k*15:15+k*15,:]
            



        edge_model = covariance.GraphLassoCV()

        X = sampleFeatureMatrix.copy()
        X /= X.std(axis=0)
        edge_model.fit(X)
        
        _, labels = cluster.affinity_propagation(edge_model.covariance_)
        n_labels = labels.max()

        
        if i==0:
            amexF1[k,:] = labels

        if i==1:
            amexF2[k,:] = labels

        if i==2:
            amexF3[k,:] = labels

        if i==3:
            amexF4[k,:] = labels

        if i==4:
            amexF5[k,:] = labels

        if i==5:
            amexF6[k,:] = labels

        if i==6:
            amexF7[k,:] = labels
        print k



AMEXdict = pickle.load(open("AMEXdictRanked.p","rb"))
NYSEdict = pickle.load(open("NYSEdictRanked.p","rb"))
NASDAQdict = pickle.load(open("NASDAQdictRanked.p","rb"))

amexN = len(AMEXdict)
nyseN = len(NYSEdict)
nasdaqN = len(NASDAQdict)

"""

1. upload the cluster number matrix using pickle
2. AMEXf1[i,j] = company j at time i 
3. transpose it so that AMEXf1[i,j] = company i at time j 

"""
########### transpose all these matrices
AMEXf1 = np.transpose( pickle.load(open("amexMomentum.p","rb")))
AMEXf2 = np.transpose( pickle.load(open("amexROC.p","rb")))
AMEXf3 = np.transpose( pickle.load(open("amexAD.p","rb")))
AMEXf4 = np.transpose( pickle.load(open("amexDisp5.p","rb")))
AMEXf5 = np.transpose( pickle.load(open("amexDisp10.p","rb")))
AMEXf6 = np.transpose( pickle.load(open("amexOSCP.p","rb")))
AMEXf7 = np.transpose( pickle.load(open("amexRSI.p","rb")))

NYSEf1 = np.transpose( pickle.load(open("nyseMomentum.p","rb")))
NYSEf2 = np.transpose( pickle.load(open("nyseROC.p","rb")))
NYSEf3 = np.transpose( pickle.load(open("nyseAD.p","rb")))
NYSEf4 = np.transpose( pickle.load(open("nyseDisp5.p","rb")))
NYSEf5 = np.transpose( pickle.load(open("nyseDisp10.p","rb")))
NYSEf6 = np.transpose( pickle.load(open("nyseOSCP.p","rb")))
NYSEf7 = np.transpose( pickle.load(open("nyseRSI.p","rb")))

NASDAQf1 = np.transpose( pickle.load(open("nasdaqMomentum.p","rb")))
NASDAQf2 = np.transpose( pickle.load(open("nasdaqROC.p","rb")))
NASDAQf3 = np.transpose( pickle.load(open("nasdaqAD.p","rb")))
NASDAQf4 = np.transpose( pickle.load(open("nasdaqDisp5.p","rb")))
NASDAQf5 = np.transpose( pickle.load(open("nasdaqDisp10.p","rb")))
NASDAQf6 = np.transpose( pickle.load(open("nasdaqOSCP.p","rb")))
NASDAQf7 = np.transpose( pickle.load(open("nasdaqRSI.p","rb")))

nyseTickers = NYSEdict.keys()
amexTickers = AMEXdict.keys()
nasdaqTickers = NASDAQdict.keys()

nyseVal = list()
amexVal = list()
nasdaqVal = list()

"""
  Stockdict = dict(zip(dash, newVal))
  
"""
for i in range(amexN):
    amexVal.append((AMEXdict[amexTickers[i]][0],AMEXdict[amexTickers[i]][1],AMEXf1[i,:],AMEXf2[i,:],AMEXf3[i,:],AMEXf4[i,:],AMEXf5[i,:],AMEXf6[i,:],AMEXf7[i,:]))
    
amexDict = dict(zip(amexTickers,amexVal))

for i in range(nyseN):
    nyseVal.append((NYSEdict[nyseTickers[i]][0],NYSEdict[nyseTickers[i]][1],NYSEf1[i,:],NYSEf2[i,:],NYSEf3[i,:],NYSEf4[i,:],NYSEf5[i,:],NYSEf6[i,:],NYSEf7[i,:]))
    
nyseDict = dict(zip(nyseTickers,nyseVal))

for i in range(nasdaqN):
    nasdaqVal.append((NASDAQdict[nasdaqTickers[i]][0],NASDAQdict[nasdaqTickers[i]][1],NASDAQf1[i,:],NASDAQf2[i,:],NASDAQf3[i,:],NASDAQf4[i,:],NASDAQf5[i,:],NASDAQf6[i,:],NASDAQf7[i,:]))
    
nasdaqDict = dict(zip(nasdaqTickers,nasdaqVal))

"""
marketDict = {ticker : name,industry,fi cluster i from 1 to 7}

marketDict['goog'][0] = name
marketDict['goog'][1] = (sector,industry)

marketDict['goog'][2 + (i-1)] = feature i cluster array  for google
so marketDict['goog'][2 + (i-1)][j] = feature i cluster number  at time interval j for google

"""

def clusterMarket(marketDict):
    
    key = marketDict.keys()
    N = len(key)
    
    weight = np.zeros((N,N))
    
    for i in range(N):
        for j in range(i+1,N):
            buffer = 0
            
            # add 1 for sector 
            if marketDict[key[i]][1][0] == marketDict[key[j]][1][0]:
                buffer += 1
            
            # add 1 for industry
            if marketDict[key[i]][1][1] == marketDict[key[j]][1][1]:
                buffer += 1
                
            arr1 = marketDict[key[i]][2]
            arr2 = marketDict[key[i]][3]
            arr3 = marketDict[key[i]][4]
            arr4 = marketDict[key[i]][5]
            arr5 = marketDict[key[i]][6]
            arr6 = marketDict[key[i]][7]
            arr7 = marketDict[key[i]][8]
            
            brr1 = marketDict[key[j]][2]
            brr2 = marketDict[key[j]][3]
            brr3 = marketDict[key[j]][4]
            brr4 = marketDict[key[j]][5]
            brr5 = marketDict[key[j]][6]
            brr6 = marketDict[key[j]][7]
            brr7 = marketDict[key[j]][8]
            
            M = len(arr1)
            bufferArr = 0
            for k in range(M):
                if arr1[k]==brr1[k]: bufferArr+=1
                if arr2[k]==brr2[k]: bufferArr+=1
                if arr3[k]==brr3[k]: bufferArr+=1
                if arr4[k]==brr4[k]: bufferArr+=1
                if arr5[k]==brr5[k]: bufferArr+=1
                if arr6[k]==brr6[k]: bufferArr+=1
                if arr7[k]==brr7[k]: bufferArr+=1
            
            bufferArr = bufferArr/3.
            buffer += bufferArr
            weight[i,j] = buffer
            
            weight[j,i] = weight[i,j]
    
    _, labels = cluster.affinity_propagation(weight)
    #n_labels = labels.max()
    
    newVal = list()
    for i in range(N):
        newVal.append((marketDict[key[i]][0],marketDict[key[i]][1],labels[i]))
        
    newDict = dict(zip(key,newVal))
    
    return newDict, weight
    
amexDICT,weightAMEX = clusterMarket(amexDict)
nyseDICT,weightNYSE = clusterMarket(nyseDict)
nasdaqDICT,weightNASDAQ = clusterMarket(nasdaqDict)

pickle.dump(amexDICT , open("amexCluster.p","wb"))
pickle.dump(nyseDICT , open("nyseCluster.p","wb"))
pickle.dump(nasdaqDICT , open("nasdaqCluster.p","wb"))

pickle.dump(weightAMEX , open("amexWEIGHT.p","wb"))
pickle.dump(weightNYSE , open("nyseWEIGHT.p","wb"))
pickle.dump(weightNASDAQ , open("nasdaqWEIGHT.p","wb"))
            
    
    

