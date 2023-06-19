
#This class was created to get all ODLs as dataframe
from pandas import *
import pandas as pd
import sqlalchemy as db

# connect to database
engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
connection = engine.connect()
metadata = db.MetaData()
# get table
odls = db.Table('odls', metadata, autoload=True, autoload_with=engine)


def getAllOdls():
    df = pd.read_sql_table( odls, con=engine)
    return(df)
