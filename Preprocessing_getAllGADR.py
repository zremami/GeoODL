
# import urllib library
# importing module
from pandas import *
import pandas as pd
import sqlalchemy as db


engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
connection = engine.connect()
metadata = db.MetaData()

odls = db.Table('odls', metadata, autoload=True, autoload_with=engine)


def getAllOdls():
    #query = db.select([odls]) 
    
    #ResultProxy = connection.execute(query)
    #ResultSet = ResultProxy.fetchall()
    df = pd.read_sql_table( odls, con=engine)
    return(df)
