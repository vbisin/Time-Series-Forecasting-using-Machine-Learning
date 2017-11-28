# We have 6 major python scripts (SVM_mode, googleIndustryCrawler.py, getData.py, Extra.py. clustering.py, and plotClusters.py) that we use for our project (we have preprocessed the data already). 

## i) SVM_model.py  (Run this file , it contains the SMV model and plots the error rates)

1) Define the 24 technical indicators for our input X and a output Y for 1,5, and 10 day projections. 

2) The variable Y is 1 if there is a positive price change and -1 if not. This price change is taken over a 1,5, or 10 day interval. By far the highest accuracy we get is for 10 day intervals. 

3) We then use the yahoo_finance module to obtain historical data from 2014-2016 (We have already stored the historical data in the SVM model folder for FedEx )

4) We then use the clustering data we have already processed to find companies in the same cluster as FedEx and get their historical data as well. Then we define the input and output using this historical data.

5) Finally we use Support Vector Machines with a polynomial kernel and Cross-Validation to find the optimal C* and d*. And we plot the training and testing error for FedEx along with printing the Test Error.
Note that if one wishes to run this for another company besides FedEx, one needs an internet connection to obtain the historical data, then we have provided a list of companies and their Tickers to input in our function called runmodel

## ii) googleIndustryCrawler.py
Contains the function industry, which is a web crawler that attains the sector and industry (as a tuple) from Google Finance.

## iii) getData.py 
In this file we import the quotes using the Yahoo! Python module for AMEX, NYSE, and NASDAQ. We remove any stocks that have incomplete data.

## iv) Extra.py 
In this file we remove any stocks from our dictionaries and quotes lists that are funds. We do so by simply iterating through their industry data. We also run Ridge, Elastic Net, and Lasso regressions to determine the most important of the 12 features (the result was always 7).

## v) clustering.py
In this file, for each stock market we cluster over the 7 features described in the Extra.py file, using a sparse inverse covariance matrix and affinity propagation. We then sum up the number of times any two stocks were in the same cluster over each iteration n defined as the weight” matrix. We then cluster over this matrix one last time to achieve the final cluster results. 

## vi) plotClusters.py 
Building on code written by Varoquaux (see works cited in the report), we plot each of the 3 stock markets. These images can be seen either in the lecture slides or the report. 


  
