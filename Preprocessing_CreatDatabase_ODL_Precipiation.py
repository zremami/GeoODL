
#This class was created to get odl and precipiaton wfs data and store as a table in databse
# import urllib library
from urllib.request import urlopen
# importing module
from pandas import *
import pandas as pd
# import json
import json
import sqlalchemy as db

#connect to database
engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
connection = engine.connect()
metadata = db.MetaData()

 
# reading CSV file for selected sations
data = read_csv("/home/raha/Raha/Thesis/Data/currently_active_odl_stations_selection.csv")
 
# converting column data to list
loc_Codes = data['locality_code'].tolist()

def CreateDatabse_ODL_Precipiation():
    # store the URL in url as 
    # parameter for urlopen
    #typeName = 'opendata:public_odl_brutto_1h'
    typeName = 'opendata:public_precipitation_15min'
    try:
        for id in loc_Codes:
            url = "https://entw-imis.lab.bfs.de/ogc/opendata/wfs?typeName="+typeName+"&_dc=1665655122146&service=WFS&version=1.1.0&request=GetFeature&outputFormat=application%2Fjson&srsname=EPSG%3A3857&cql_filter=id%20%3D%20%27"+id+"%27%20AND%20end_measure%3E%272021-01-01T00%3A00%3A00.000Z%27%20AND%20end_measure%3C%272022-12-31T00%3A00%3A00.000Z%27"
        
            # store the response of URL
            response = urlopen(url)       
            # storing the JSON response 
            # from url in data
            data_json = json.loads(response.read())
            datanorm= pd.json_normalize(data_json,"features")
            df = pd.DataFrame(datanorm)
            
            if df.empty:
                continue
            if typeName == 'opendata:public_odl_brutto_1h':
                pssql_table = "odls"
                df=df[['properties.id','properties.start','properties.end_measure','properties.value']]
                df.columns = ['Locality_code', 'Start_measure','End_measure','Value']
            elif typeName == 'opendata:public_precipitation_15min':
                pssql_table = "precipitations"
                df = df[['properties.id','properties.start_measure','properties.end_measure','properties.value']]
                df.columns = ['Locality_code', 'Start_measure','End_measure','Value']

            # save in a databse
            df.to_sql(name=pssql_table, con=connection, if_exists='append',index=False, method='multi')
    except:
        print("An exception occurred")

CreateDatabse_ODL_Precipiation()