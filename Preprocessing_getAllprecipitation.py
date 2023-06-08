

# import urllib library
# importing module
from pandas import *
import pandas as pd
import sqlalchemy as db

engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
connection = engine.connect()
metadata = db.MetaData()
precipitations = db.Table('precipitations', metadata, autoload=True, autoload_with=engine)


def getAllprecipitations():
    #query = db.select([odls]) 
    
    #ResultProxy = connection.execute(query)
    #ResultSet = ResultProxy.fetchall()
    df = table_df = pd.read_sql_table( precipitations, con=engine)
    return df