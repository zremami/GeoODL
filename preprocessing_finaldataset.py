from pandas import *
def DataFrameModelModel():
    from Preprocessing_getAllGADR import getAllOdls
    from Preprocessing_getAllprecipitation import getAllprecipitations
    import pandas as pd
    import numpy as np
    import sqlalchemy as db
    
    engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
    connection = engine.connect()
    metadata = db.MetaData()
    # get all odls
    df1=getAllOdls()
    df1 = pd.DataFrame(df1)

    # convert to datetime 
    df1['Start_measure'] = pd.to_datetime(df1['Start_measure']).dt.tz_localize(None)
    df1['End_measure'] = pd.to_datetime(df1['End_measure']).dt.tz_localize(None)
    df1['Value'] = df1['Value'].astype(float)

    # create a list with index time 
    df1 = df1.squeeze()
    df1.set_index('Start_measure', inplace=True)

    # get 15min data
    df2=getAllprecipitations()
    df2 = pd.DataFrame(df2)

    # convert to datetime 
    df2['Start_measure'] = pd.to_datetime(df2['Start_measure']).dt.tz_localize(None)
    df2['End_measure'] = pd.to_datetime(df2['End_measure']).dt.tz_localize(None)
    df2['Value'] = df2['Value'].astype(float)


    # create a list to resample and agregate data
    df2 = df2.squeeze()
    df2.set_index('Start_measure', inplace=True)
    # resample it
    
    
    data = read_csv("/home/raha/Raha/Thesis/Data/currently_active_odl_stations_selection.csv")
    loc_Codes = data['locality_code'].tolist()
    # considering data for each location
    for i in loc_Codes:
        df1_locality_code = df1.loc[df1['Locality_code'] == i]
        df2_locality_code = df2.loc[df2['Locality_code'] == i]

        if df1_locality_code.empty == True:
            continue

        if df2_locality_code.empty == True:
            continue

        df2_locality_code=df2_locality_code.resample('H')['Value'].mean()
        #print(df2_locality_code.head())

        df3 = df1_locality_code.join(df2_locality_code, lsuffix='_odl', rsuffix='_precipitation', how='inner')
        #print(df3)

        # drop all rows with any NaN and NaT values
        df3 = df3.dropna()
        #print(df3)

        # create list of precipitationMinus2
        value_precipitationMinus2List = []
        for index in range(len(df3)):
            #if index == len(df3)- 2:
                #break
            if index == 0 or index == 1:
                continue
            mines2=index-2
            value_precipitationMinus2List.append(df3['Value_precipitation'][mines2])
            
        df3 = df3.iloc[2:]
        #value_odlPlus2array = np.array(value_odlPlus2List)
        df3['Value_precipitationMinus2'] = np.array(value_precipitationMinus2List)
        df3['Month'] = df3['End_measure'].dt.month 
        df_dummies = pd.get_dummies(df3,columns=['Month'])
        pssql_table = "odls_precipitations_dummy"
        df_dummies.to_sql(name=pssql_table, con=connection, if_exists='append')

DataFrameModelModel()