"""
Created on Fri Dec  2 23:58:01 2016

@author: Vitto
"""
import datetime
import numpy as np
from matplotlib.finance import quotes_historical_yahoo_ochl
import pickle

d1 = datetime.datetime(2010, 1, 1)
d2 = datetime.datetime(2016, 1, 1)
NYSEdict={}
NASDAQdict={}
AMEXdict={}
NYSEquotes=list()
NASDAQquotes=list()
AMEXquotes=list()
from yahoo_finance import Share
from googleIndustryCrawler import industry

tickers=np.loadtxt('companylistNYSE.txt',dtype='U15',delimiter=',',skiprows=1,usecols=[0])
bam=0

for ticker in tickers:
    #remove quotation marks around ticker 
    bam+=1
    print(bam)
    ticker = ticker[1:-1]
    error=0
    #make sure it really represents a stock
    
    try:    
        yahooSymbol=Share(ticker)
    except:
        print('error in starting ticker- ' + ticker + " deleted")
        error=1
    try:    
        yahooIndustry=industry(ticker)  
    except:
        print('error in getting industry- ' + ticker + " deleted")
        error=1
    
    if error!=1:
        # If NYSE
        if str(yahooSymbol.get_stock_exchange()) == "NYQ":
            NYSEdict[ticker]=(yahooSymbol.get_name(),yahooIndustry)
            current=len(NYSEquotes)
            try:
                NYSEquotes.append(quotes_historical_yahoo_ochl(ticker, d1, d2, asobject=True))
            except:
                   print('error - ' + ticker + " deleted")
                   NYSEdict.pop(ticker)
                   if current != len(NYSEquotes):
                       del NYSEquotes[len(NYSEquotes)-1]
            else:
                if NYSEquotes[len(NYSEquotes)-1]==None:
                    del NYSEquotes[len(NYSEquotes)-1]
                    NYSEdict.pop(ticker)
                    print(ticker+" had attribute none")
                
                elif len(NYSEquotes[len(NYSEquotes)-1].open)!=1510:
                    del NYSEquotes[len(NYSEquotes)-1]
                    NYSEdict.pop(ticker)
                    print('length of ' + ticker + ' problem (NYSE)')
                
    
        # If AMEX
        elif str(yahooSymbol.get_stock_exchange()) == "ASE": 
            AMEXdict[ticker]=(yahooSymbol.get_name(),yahooIndustry)
            current=len(AMEXquotes)
    
            try:
                AMEXquotes.append(quotes_historical_yahoo_ochl(ticker, d1, d2, asobject=True))
            except:
                print('error - ' + ticker + " deleted")
                AMEXdict.pop(ticker)
                if current != len(AMEXquotes):
                       del AMEXquotes[len(AMEXquotes)-1]
            else:
                if AMEXquotes[len(AMEXquotes)-1]==None:
                    del AMEXquotes[len(AMEXquotes)-1]
                    AMEXdict.pop(ticker)  
                    print(ticker+" had attribute none")
                
                elif len(AMEXquotes[len(AMEXquotes)-1].open)!=1510:
                    del AMEXquotes[len(AMEXquotes)-1]
                    AMEXdict.pop(ticker)
                    print('length of ' + ticker + ' problem (AMEX)')
        
        # If NASDAQ
        elif str(yahooSymbol.get_stock_exchange()) == 'NMS' or  str(yahooSymbol.get_stock_exchange()) == 'NGM' or str(yahooSymbol.get_stock_exchange())== 'NCM':
            NASDAQdict[ticker]=(yahooSymbol.get_name(),yahooIndustry)
            current=len(NASDAQquotes)
            
            try:
                NASDAQquotes.append(quotes_historical_yahoo_ochl(ticker, d1, d2, asobject=True))
            except:
                   print('error - ' + ticker + " deleted")
                   NASDAQdict.pop(ticker)
                   if current != len(NASDAQquotes):
                       del NASDAQquotes[len(NASDAQquotes)-1]
            else: 
                if NASDAQquotes[len(NASDAQquotes)-1]==None:
                    del NASDAQquotes[len(NASDAQquotes)-1]
                    NASDAQdict.pop(ticker)
                    print(ticker+" had attribute none")
                    
                elif len(NASDAQquotes[len(NASDAQquotes)-1].open)!=1510:
                     del NASDAQquotes[len(NASDAQquotes)-1]
                     NASDAQdict.pop(ticker)
                     print('length of ' + ticker + ' problem (NASDAQ)')
                     
        # Not from any  of our stock exchanges 
        elif str(yahooSymbol.get_stock_exchange()) != "NGM" or str(yahooSymbol.get_stock_exchange()) != "ASE" or str(yahooSymbol.get_stock_exchange()) != "NMS" or str(yahooSymbol.get_stock_exchange()) != "NGM":
            if  error!=1:
                print(ticker+" deleted because not in any of our stock exchanges (i.e. "+str(yahooSymbol.get_stock_exchange())+")" )


# save dictionaries 
pickle.dump(NYSEdict, open("NYSEdict.p", "wb"))
#pickle.dump(NASDAQdict, open("NASDAQdict.p", "wb"))
#pickle.dump(AMEXdict, open("AMEXdict.p", "wb"))

# save quotes
pickle.dump(NYSEquotes, open("NYSEquotes.p", "wb"))
#pickle.dump(NASDAQquotes, open("NASDAQquotes.p", "wb"))
#pickle.dump(AMEXquotes, open("AMEXquotes.p", "wb"))