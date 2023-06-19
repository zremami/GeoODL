#This class was cteated to save data of traned regression model
from pandas import *
def PMultiLinearRegression_Trained():
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    import sqlalchemy as db
        
    # get csv for add geometry information
    sondenstandorte = read_csv("/home/raha/Raha/Thesis/Data/odl_sondenstandorte.csv")
    df_sondenstandorte= pd.DataFrame(sondenstandorte)
    df_sondenstandorte['locality_code']=df_sondenstandorte['locality_code'].astype("string")

    # connect to database
    engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
    connection = engine.connect()
    metadata = db.MetaData()

    #get table
    odls_precipitations = db.Table('odls_precipitations_dummy', metadata, autoload=True, autoload_with=engine)

    # convert to dataframe
    df =pd.read_sql_table( odls_precipitations, con=engine)

    # group by
    df_gropby= df.groupby('Locality_code').mean()


    for i in df_gropby.index:
        
        df_M = pd.DataFrame()

        df_locality= df.loc[df['Locality_code']== i]


        df_sondenstandorte_locality= df_sondenstandorte.loc[df_sondenstandorte['locality_code']== i]


        x = df_locality[['Value_precipitation', 'Value_precipitationMinus2','Month_1','Month_2','Month_3','Month_4','Month_5','Month_6','Month_7','Month_8','Month_9','Month_10','Month_11','Month_12']]
        y = df_locality['Value_odl']


       
        # linearRegression
        x = sm.add_constant(x) # adding a constant

        model = sm.OLS(y,x)

        result = model.fit()

        # extract coefficinet parameters
        coefficients = result.params
        coefArray = np.array(coefficients)

        # save coefficinet parameters and other information
        df_M['Locality_code']=df_sondenstandorte_locality['locality_code']
        df_M['Locality_name']=df_sondenstandorte_locality['locality_name']
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
        df_M['latitude']=np.array(df_sondenstandorte_locality['latitude'])
        df_M['longitude']=np.array(df_sondenstandorte_locality['longitude'])

        # save in database
        pssql_table = "PMultiLinearRegression_Trained"
        df_M.to_sql(name=pssql_table, con=connection, if_exists='append',index=False)

 
PMultiLinearRegression_Trained()