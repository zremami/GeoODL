

#This class was created to get all precipiation data from data base as dataframe
from pandas import *
import pandas as pd
import sqlalchemy as db

#connect to database
engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
connection = engine.connect()
metadata = db.MetaData()
# get table
precipitations = db.Table('precipitations', metadata, autoload=True, autoload_with=engine)


def getAllprecipitations():
    df = table_df = pd.read_sql_table( precipitations, con=engine)
    return df