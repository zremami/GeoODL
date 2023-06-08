from pandas import *
def MultiLinearRegressionTrain():
    import pandas as pd
    import numpy as np
    import sklearn 
    from sklearn.linear_model import LinearRegression
    import random
    from scipy import stats
    import statsmodels.api as sm
    from sklearn import linear_model
    import sqlalchemy as db
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score



    sondenstandorte = read_csv("/home/raha/Raha/Thesis/Data/odl_sondenstandorte.csv")
    df_sondenstandorte= pd.DataFrame(sondenstandorte)
    df_sondenstandorte['locality_code']=df_sondenstandorte['locality_code'].astype("string")

    #df_sondenstandorte['locality_code']=df_sondenstandorte['locality_code'].str.strip()

    engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
    connection = engine.connect()
    metadata = db.MetaData()
    odls_precipitations = db.Table('odls_precipitations_dummy', metadata, autoload=True, autoload_with=engine)

    df =pd.read_sql_table( odls_precipitations, con=engine)
    df_gropby= df.groupby('Locality_code').mean()


    for i in df_gropby.index:
        
        df_M = pd.DataFrame()

        df_locality= df.loc[df['Locality_code']== i]


        df_sondenstandorte_locality= df_sondenstandorte.loc[df_sondenstandorte['locality_code']== i]

        x = df_locality[['Value_precipitation', 'Value_precipitationMinus2','Month_1','Month_2','Month_3','Month_4','Month_5','Month_6','Month_7','Month_8','Month_9','Month_10','Month_11','Month_12']]
        y = df_locality['Value_odl']

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

       

        model = sm.OLS(y_train,X_train)

        result = model.fit()


        prediction = result.predict(X_test)

        MAE = mean_absolute_error(np.array(y_test), np.array(prediction))

        df_M['Locality_code']=df_sondenstandorte_locality['locality_code']
        df_M['Locality_name']=df_sondenstandorte_locality['locality_name']
        df_M['MAE']=np.array(MAE)

        pssql_table = "PMuliLinearRegression_Tested"
        df_M.to_sql(name=pssql_table, con=connection, if_exists='append')

 
MultiLinearRegressionTrain()