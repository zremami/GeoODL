#This class was created to tain the data as see OLS table for the trained data for each station
from pandas import *
def PMultiLinearRegression_train_Test_Fly():
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import sqlalchemy as db

        

    # connect to database
    engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
    connection = engine.connect()
    metadata = db.MetaData()
    # get table
    odls_precipitations = db.Table('odls_precipitations_dummy', metadata, autoload=True, autoload_with=engine)

    # create data farme
    df =pd.read_sql_table( odls_precipitations, con=engine)
    df_gropby= df.groupby('Locality_code').mean()

    i="DEZ3551"
    df_M = pd.DataFrame()
    df_locality= df.loc[df['Locality_code']== i]

    # dependent and indipentent variables
    x = df_locality[['Value_precipitation', 'Value_precipitationMinus2','Month_1','Month_2','Month_3','Month_4','Month_5','Month_6','Month_7','Month_8','Month_9','Month_10','Month_11','Month_12']]
    y = df_locality['Value_odl']

    # with statsmodels
    X = sm.add_constant(x) # adding a constant
    model = sm.OLS(y, X).fit()

    # print result
    print_model = model.summary()
    print(print_model)
    print(i)

 
PMultiLinearRegression_train_Test_Fly()