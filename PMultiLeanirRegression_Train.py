from pandas import *
def PMultiLinearRegression_train_Test_Fly():
    import pandas as pd
    import numpy as np
    from sklearn import linear_model
    import statsmodels.api as sm
    import random
    from scipy import stats
    import sqlalchemy as db
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    import json
    from urllib.request import urlopen
    from PDtaFrame_7days_Clean import DataFrame7daysModel
        


    engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
    connection = engine.connect()
    metadata = db.MetaData()
    #odls_precipitations_dummy
    #odls_precipitations
    odls_precipitations = db.Table('odls_precipitations_dummy', metadata, autoload=True, autoload_with=engine)

    df =pd.read_sql_table( odls_precipitations, con=engine)
    
    i="DEZ0091"
    df_M = pd.DataFrame()
    df_locality= df.loc[df['Locality_code']== i]

    x = df_locality[['Value_precipitation', 'Value_precipitationMinus2','Month_1','Month_2','Month_3','Month_4','Month_5','Month_6','Month_7','Month_8','Month_9','Month_10','Month_11','Month_12']]
    #x = df_locality[['Value_precipitation', 'Value_precipitationMinus2']]
    y = df_locality['Value_odl']


    #df_locality = pd.get_dummies(df_locality,columns=['Month'])


    #duplicatef2 = df_locality[df_locality['Start_measure'].duplicated()]
    #print(duplicatef2)


    #isnulldf = df.isna()
    #print(isnulldf.sum())



    # with statsmodels
    X = sm.add_constant(x) # adding a constant

    model = sm.OLS(y, X).fit()

    print_model = model.summary()
    print(print_model)

    coefficients = model.params

    # Print the coefficients
    print(coefficients)

 
PMultiLinearRegression_train_Test_Fly()