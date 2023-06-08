from pandas import *
import pandas as pd
import numpy as np
import sqlalchemy as db
def DataFrame7daysModel(df1,df2):
    
    # convert to datetime 
    df1['Start_measure'] = pd.to_datetime(df1['Start_measure']).dt.tz_localize(None)
    df1['End_measure'] = pd.to_datetime(df1['End_measure']).dt.tz_localize(None)
    df1['Value'] = df1['Value'].astype(float)

    # create a list to resample and agregate data
    df1 = df1.set_index(df1['End_measure'])

    # convert to datetime 
    df2['Start_measure'] = pd.to_datetime(df2['Start_measure']).dt.tz_localize(None)
    df2['End_measure'] = pd.to_datetime(df2['End_measure']).dt.tz_localize(None)
    df2['Value'] = df2['Value'].astype(float)


    # create a list to resample and agregate data
    df2 = df2.set_index(df2['End_measure'])

    # resample it
    
    df2=df2.resample('H').mean()
    #print(df2_locality_code.head())


    df3 = df1.join(df2, lsuffix='_odl', rsuffix='_precipitation', how='inner')
    #print(df3)

    # drop all rows with any NaN and NaT values
    df3 = df3.dropna()
    #print(df3)

    value_precipitationMinus2List = []
    # i++ = (1,len(df3),2)
    for index in range(len(df3)):
        #if index == len(df3)- 2:
            #break
        if index == 0 or index == 1 or index == 2:
            continue
        mines2=index-2
        value_precipitationMinus2List.append(df3['Value_precipitation'][mines2])
        
    df3 = df3.iloc[:-3]

    #value_odlPlus2array = np.array(value_odlPlus2List)
    df3['Value_precipitationMinus2'] = np.array(value_precipitationMinus2List)
    df3['Month'] = df3['End_measure'].dt.month
    df_dummies = pd.get_dummies(df3,columns=['Month']) 

    return df_dummies