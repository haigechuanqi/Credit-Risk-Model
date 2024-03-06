import pandas as pd
card_data = pd.read_excel("C:\\Users\\Hai ge chuan qi\\CRA\\card.xlsx")
CRE_data = pd.read_excel("C:\\Users\\Hai ge chuan qi\\CRA\\CRE.xlsx")
card_data = card_data.drop_duplicates()#drop duplicates
CRE_data = CRE_data.drop_duplicates()
#card_data.head()
CRE_data.head()

from statsmodels.tsa.stattools import adfuller
card_data['card_chargeoffs'] = card_data['chargeoffs']/card_data['loans']
CRE_data['CRE_chargeoffs'] = CRE_data['chargeoffs']/CRE_data['loans']
card_data = card_data.drop('chargeoffs', axis =1)
card_data = card_data.drop('loans', axis = 1)
CRE_data = CRE_data.drop('loans', axis=1)
CRE_data = CRE_data.drop('chargeoffs', axis =1)
df1 = pd.merge(CRE_data, card_data, on = 'date', how = 'left')
print(df1.shape)
print(CRE_data.shape)
print(card_data.shape)
#confirmed no observations dropped.


adft_CRE = 100*adfuller(df1.CRE_chargeoffs)[1] #Performing Adfuller test for CRE_chargeoffs
adfuller1 = f'{adft_CRE:3.2f}'
if adft_CRE < 5:
    print(f'\nThe AdFuller for CRE Chargeoffs is {adfuller1}%, and its stationary')
else:
    print(f'\nThe AdFuller for CRE Chargeoffs is {adfuller1}%, and its unstationary')
adft_card= 100*adfuller(df1.card_chargeoffs)[1] #Performing Adfuller test for card_chargeoffs
adfuller2 = f'{adft_card:3.2f}'
if adft_card < 5:
    print(f'\nThe AdFuller for card Chargeoffs is {adfuller2}%, and its stationary')
else:
    print(f'\nThe AdFuller for card Chargeoffs is {adfuller2}%, and its unstationary')

#Take the first difference to CRE and card chargeoffs, since they aren't stationary
df1['diff_CRE'] = df1['CRE_chargeoffs']-df1['CRE_chargeoffs'].shift()
df1['diff_card'] = df1['card_chargeoffs'] - df1['card_chargeoffs'].shift()
df1 = df1[~df1.diff_card.isna()]
df1.head()
#Test for stationarity
print(f'\nThe ADFuller is {100*adfuller(df1.diff_CRE)[1]:3.2f}%')
print(f'\nThe ADFuller is {100*adfuller(df1.diff_card)[1]:3.2f}%')

import yfinance as yf
import pandas_datareader.data as web
import pandas as pd
import datetime as dt


u3 = web.DataReader("UNRATE", "fred", start = '2000-01-01') #Retrieving Unemployement rate data
u3.plot(title="US inflation", legend=False)
u3 = u3['UNRATE']/100
u3 = pd.DataFrame(u3)
type(u3)

#Unemployment
import datetime as dt
from statsmodels.tsa.stattools import adfuller
#Adjusting the frequency to quarterly, and adjust the date to one day forwawrd
u3['date'] = [x.date() - dt.timedelta(days=1) for x in u3.index]
u3 = u3[u3.date <= dt.date(2020,1,1)]
u3['month'] = [x.month for x in u3.date]
u3['year'] = [x.year for x in u3.date]
u3 = u3[u3.month.isin([3,6,9,12])]
print(u3)
adfullertest = 100*adfuller(u3.UNRATE)[1]
adfuller_0 = f'{adfullertest:3.2f}'
if adfullertest < 5:
    print(f'\nThe AdFuller is {adfuller_0}%, and its stationary')
else:
    print(f'\nThe AdFuller is {adfuller_0}%, and its unstationary')
type(u3.UNRATE)

# #Take the first difference to Unemployment rate, since it is unstationary
u3['u3_lag'] = u3.UNRATE.shift()
u3['dur'] = u3.UNRATE - u3.u3_lag
u3['dur_lag'] = u3.dur.shift()
#u3 = u3.fillna(method = 'ffill')
u3 = u3.dropna()
u3.shape
type(u3.dur_lag)
#u3.dur_lag
tst = u3.dur_lag
u3.head()
#type(tst)
print(f'\nThe ADFuller is {100*adfuller(u3.dur_lag)[1]:3.2f}%')

#Oil Price
oil_price = web.DataReader('DCOILBRENTEU', 'fred', start = '2000-01-01')
oil_price.plot(title = 'Oil price', legend = False)
#print(oil_price)
print(oil_price.iloc[60:70])
#Adjusting the frequency to quarterly, and adjust the date to one day forwawrd
oil_price['date'] = [x.date() for x in oil_price.index]
oil_price = oil_price.resample('Q').last()
oil_price = oil_price[oil_price.date <= dt.date(2020,1,1)]
oil_price = oil_price.fillna(method = 'ffill')
oil_price = pd.DataFrame(oil_price)
print(oil_price)
print(f'\nThe ADFuller is {100*adfuller(oil_price.DCOILBRENTEU)[1]:3.2f}%')
#Take the first difference to oil_price, since it is unstationary
oil_price2 = oil_price 
print(type(oil_price))
oil_price2['oil_lag'] = oil_price2.DCOILBRENTEU.shift()
oil_price2['doil'] = oil_price2.DCOILBRENTEU - oil_price.oil_lag
oil_price2['doil_lag'] = oil_price2.doil.shift()
oil_price2 = oil_price2.fillna(method = 'ffill')
#oil_price2 = oil_price2.drop(pd.to_datetime(['2000-03-31', '2000-06-30']))
oil_price2['date'] = [x.date() for x in oil_price2.index] #adjustment to make the date as last day of quarter

oil_price3 = oil_price2.dropna()
#print(oil_price)
print(f'\nThe ADFuller is {100*adfuller(oil_price3.doil_lag)[1]:3.2f}%')
#oil_price2.head()
#oil_price.shape

#Treasury
Treasury_rate = web.DataReader('T10Y2Y', 'fred', start = '2000-01-01')
Treasury_rate.plot(title = '10-year minus 2-year treasury rates', legend = False)
print(Treasury_rate)
#Adjusting the frequency to quarterly, and adjust the date to one day forwawrd
Treasury_rate['date'] = [x.date() for x in Treasury_rate.index]
Treasury_rate = Treasury_rate.resample('Q').last()
Treasury_rate = Treasury_rate[Treasury_rate.date <= dt.date(2020,1,1)]
Treasury_rate = Treasury_rate.fillna(method = 'ffill')
print(Treasury_rate)
print(f'\nThe ADFuller is {100*adfuller(Treasury_rate.T10Y2Y)[1]:3.2f}%')
#Take the first difference to oil_price, since it is unstationary
Treasury_rate['TR_lag'] = Treasury_rate.T10Y2Y.shift()
Treasury_rate['dTR'] = Treasury_rate.T10Y2Y - Treasury_rate.TR_lag
Treasury_rate['dTR_lag'] = Treasury_rate.dTR.shift()
Treasury_rate = Treasury_rate.fillna(method = 'ffill')
Treasury_rate = Treasury_rate[~Treasury_rate.dTR_lag.isna()]
Treasury_rate['date'] = [x.date() for x in Treasury_rate.index]
print(Treasury_rate)
print(f'\nThe ADFuller is {100*adfuller(Treasury_rate.dTR_lag)[1]:3.2f}%')

#Vol Index
Vol_index = web.DataReader('VIXCLS', 'fred', start = '2000-01-01')
Vol_index.plot(title = 'CBOE Volatility Index', legend = False)
print(Vol_index)
#Adjusting the frequency to quarterly, and adjust the date to one day forwawrd
Vol_index['date'] = [x.date() for x in Vol_index.index]
Vol_index = Vol_index[Vol_index.date <= dt.date(2020,1,1)]
Vol_index = Vol_index.resample('Q').last()
Vol_index['VIXCLS'] = Vol_index.VIXCLS.shift()
Vol_index = Vol_index.fillna(method = 'ffill')
Vol_index = Vol_index[~Vol_index.VIXCLS.isna()]
Vol_index['date'] = [x.date() for x in Vol_index.index]
print(Vol_index)
print(f'\nThe ADFuller is {100*adfuller(Vol_index.VIXCLS)[1]:3.2f}%') #It is stationary

#GDP
gdp = web.DataReader("GDP", "fred", start = '2000-01-01')
gdp.plot(title = 'GDP', legend = False)
gdp_growth = gdp['GDP'].pct_change()
gdp_growth = pd.DataFrame(gdp_growth)
gdp_growth['date'] = [x.date() - dt.timedelta(days=1) for x in gdp_growth.index]
gdp_growth = gdp_growth[gdp_growth.date <= dt.date(2020,1,1)]
gdp_growth = gdp_growth.rename(columns ={'GDP': "GDP_GROWTH"})
gdp_growth = gdp_growth.fillna(method = 'ffill')
gdp_growth = gdp_growth[~gdp_growth.GDP_GROWTH.isna()]
print(gdp_growth)
print(f'\nThe ADFuller is {100*adfuller(gdp_growth.GDP_GROWTH)[1]:3.2f}%')# It's stationary


#Merge dataframe
df2 = pd.merge(gdp_growth, u3[['date', 'dur_lag']], on='date')
df2 = pd.merge(df2, oil_price3[['date', 'doil_lag']], on='date')
df2 = pd.merge(df2, Treasury_rate[['date' ,'dTR_lag']], on = 'date')
df2 = pd.merge(df2, Vol_index[['date' ,'VIXCLS']], on = 'date')
df2 = df2.drop([0,1])
df2 = df2.reset_index(drop=True)
df2 = df2.rename(columns={'date': 'date1'})
df1.head()

df3 = pd.concat([df1, df2],axis = 1)
df3 = df3.fillna(method = 'ffill')#fill the missing data with the nearest available one
na = df3.isna()#Test for N/A
df3 = df3.reset_index(drop=True)
df3 = df3.drop('date1', axis = 1)
df3['lag_CRE'] = df3.CRE_chargeoffs.shift()
df3['lag_card'] = df3.card_chargeoffs.shift()
df3 = df3[~df3.lag_CRE.isna()]

#OLS for Time Series
from itertools import combinations
import statsmodels.api as sm
list1 = ['lag_CRE', 'GDP_GROWTH', 'dur_lag', 'doil_lag', 'dTR_lag', 'VIXCLS']
list2 = ['lag_card', 'GDP_GROWTH', 'dur_lag', 'doil_lag', 'dTR_lag', 'VIXCLS']
#Model to predict CRE_chargeoffs
factor_combs = list(combinations(list1,3))
Y = df3.CRE_chargeoffs
best_r_squared = -1
best_combination = None
for i in range(0,len(factor_combs)):
    combination = factor_combs[i]
    X = df3[list(combination)]
    X = sm.add_constant(X)
    test = sm.OLS(Y,X).fit()
    r_squared = test.rsquared
    if r_squared > best_r_squared:
        best_r_squared = r_squared
        best_combination = combination
    else: pass
print("Best combination of predictor variables:", best_combination)
print("Best R-squared value:", best_r_squared)
X1 = df3[list(best_combination)]
X1 = sm.add_constant(X1)
model1 = sm.OLS(Y,X1).fit()
print(model1.summary())

#Model to predict card_chargeoffs
factor_combs1 = list(combinations(list2,3))
Y = df3.card_chargeoffs
best_r_squared1 = -1
best_combination1 = None
for i in range(0,len(factor_combs1)):
    combination = factor_combs1[i]
    X = df3[list(combination)]
    X = sm.add_constant(X)
    test = sm.OLS(Y,X).fit()
    r_squared = test.rsquared
    if r_squared > best_r_squared1:
        best_r_squared1 = r_squared
        best_combination1 = combination
    else: pass
print("Best combination of predictor variables:", best_combination1)
print("Best R-squared value:", best_r_squared1)
X2 = df3[list(best_combination1)]
X2 = sm.add_constant(X2)
model2 = sm.OLS(Y,X2).fit()
print(model2.summary())