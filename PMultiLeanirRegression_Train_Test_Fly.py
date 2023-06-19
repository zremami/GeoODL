#This class was created to train and test data via WFS 7 days data to be used later as csv for visualization
from pandas import *
def PMultiLinearRegression_train_Test_Fly():
    import pandas as pd
    import numpy as np
    import sklearn 
    import random
    import statsmodels.api as sm
    import sqlalchemy as db
    import json
    from urllib.request import urlopen
    from PDtaFrame_7days_Clean import DataFrame7daysModel
        

    # connect to database
    engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
    connection = engine.connect()
    metadata = db.MetaData()

    # get the table
    odls_precipitations = db.Table('odls_precipitations_dummy', metadata, autoload=True, autoload_with=engine)
    # convert to data frame
    df =pd.read_sql_table( odls_precipitations, con=engine)
    df_gropby= df.groupby('Locality_code').mean()

    # filter by locality code
    for i in df_gropby.index:
        df_M = pd.DataFrame()
        #filter by locality code
        df_locality= df.loc[df['Locality_code']== i]

        x = df_locality[['Value_precipitation', 'Value_precipitationMinus2','Month_1','Month_2','Month_3','Month_4','Month_5','Month_6','Month_7','Month_8','Month_9','Month_10','Month_11','Month_12']]
        y = df_locality['Value_odl']

       
        # train 
        X = sm.add_constant(x) # adding a constant

        model = sm.OLS(y, X).fit()

        # coefficinet parameters
        coefficients = model.params
        coefArray = np.array(coefficients)
        # save coefficient parameters and other data
        df_M['Locality_code']=[str(i)]
        #df_M['R-squared']=[regr.score(x, y)]
        df_M['b0'] = [coefArray[0]]
        df_M['b_Precipitation']=[coefArray[1]]
        df_M['b_PrecipitationMinus2']=[coefArray[2]]
        df_M['b_Month1']=[coefArray[3]]
        df_M['b_Month2']=[coefArray[4]]
        df_M['b_Month3']=[coefArray[5]]
        df_M['b_Month4']=[coefArray[6]]
        df_M['b_Month5']=[coefArray[7]]
        df_M['b_Month6']=[coefArray[8]]
        df_M['b_Month7']=[coefArray[9]]
        df_M['b_Month8']=[coefArray[10]]
        df_M['b_Month9']=[coefArray[11]]
        df_M['b_Month10']=[coefArray[12]]
        df_M['b_Month11']=[coefArray[13]]
        df_M['b_Month12']=[coefArray[14]]


        # get odl 7 days for test
        url1 = "https://entw-imis.lab.bfs.de/ogc/opendata/wfs?typeName=opendata:public_odl_brutto_1h&_dc=1665655122146&service=WFS&version=1.1.0&request=GetFeature&outputFormat=application%2Fjson&srsname=EPSG%3A3857&cql_filter=id%20%3D%20%27"+i+"%27%20AND%20end_measure%3E%272023-01-01T00%3A00%3A00.000Z%27%20AND%20end_measure%3C%272023-01-07T00%3A00%3A00.000Z%27"
        
        # store the response of URL
        response1 = urlopen(url1)
        data_json1 = json.loads(response1.read())
        datanorm1= pd.json_normalize(data_json1,"features")
        df1 = pd.DataFrame(datanorm1)

        #get precipiation 7 dats for test
        url2 = "https://entw-imis.lab.bfs.de/ogc/opendata/wfs?typeName=opendata:public_precipitation_15min&_dc=1665655122146&service=WFS&version=1.1.0&request=GetFeature&outputFormat=application%2Fjson&srsname=EPSG%3A3857&cql_filter=id%20%3D%20%27"+i+"%27%20AND%20end_measure%3E%272023-01-01T00%3A00%3A00.000Z%27%20AND%20end_measure%3C%272023-01-07T00%3A00%3A00.000Z%27"
            
        # store the response of URL
        response2 = urlopen(url2)
        data_json2 = json.loads(response2.read())
        datanorm2= pd.json_normalize(data_json2,"features")
        df2 = pd.DataFrame(datanorm2)

        # create data farme odls
        df1 = df1[['properties.id','properties.start','properties.end_measure','properties.value']]
        df1.columns = ['Locality_code', 'Start_measure','End_measure','Value']

        #create  dataframe precipitation
        df2 = df2[['properties.id','properties.start_measure','properties.end_measure','properties.value']]
        df2.columns = ['Locality_code', 'Start_measure','End_measure','Value']

        # resample, clean, aggregare
        df_7days = DataFrame7daysModel(df1,df2)

        # create final dataset for testing
        x_test = df_7days[['Value_precipitation', 'Value_precipitationMinus2','Value_odl']]
        y_prediction=[]
        y_real=[]

        # put in formula
        for index in range(len(df_7days)):
            y_prediction.append(df_M['b0']+df_7days['Value_precipitation'][index]*df_M['b_Precipitation'] + df_7days['Value_precipitationMinus2'][index]*df_M['b_PrecipitationMinus2'] + (1*df_M['b_Month1']))


        # save result
        y_real = np.array(df_7days['Value_odl'])
        df_final = pd.DataFrame()
        df_final['Locality_code']= np.array(df_7days['Locality_code'])
        df_final['Start_measure']= np.array(df_7days['Start_measure'])
        df_final['y_ODL_real'] = y_real
        df_final['y_ODL_prediction'] = np.array(y_prediction).astype(float)


        # save in database
        pssql_table = "PMultiLinearRegression_Tested_Fly"
        df_final.to_sql(name=pssql_table, con=connection, if_exists='append',index=False)

 
PMultiLinearRegression_train_Test_Fly()