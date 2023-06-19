# This class was created to get all joined odls and precipitation
from pandas import *
import pandas as pd
import sqlalchemy as db
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

# connect to database
engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
conn = engine.connect()
metadata = db.MetaData()
# get table
odls_precipitations = db.Table('odls_precipitations_dummy', metadata, autoload=True, autoload_with=engine)

def getOdl_precipitations():
    '''odl_precipitations = db.Table('odls_precipitations', metadata,
                db.Column('Start_measure', db.DateTime()),
                db.Column('ID', db.BIGINT),
                db.Column('Locality_code', db.String(255)),
                db.Column('End_measure', db.DateTime()),
                db.Column('Value_odl', db.Float),
                db.Column('Value_precipitation', db.Float),
                db.Column('Value_precipitationMinus2', db.Float),
                db.Column('Month', db.BIGINT)
                )

    metadata.create_all(engine) 

    output = conn.execute("SELECT * FROM odls WHERE 'odls.Locality_code' = 'DEZ3159'")
    print(output.fetchall())'''

    df = table_df = pd.read_sql_table(odls_precipitations, con=engine)
    
    return(df)

getOdl_precipitations()

