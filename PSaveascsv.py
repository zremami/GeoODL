# This class was created to save data frame to csv
from pandas import *
import pandas as pd
import sqlalchemy as db
from pathlib import Path  

#connect to datbase
engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
connection = engine.connect()
metadata = db.MetaData()
# get the table
MultiLinearRegression_Test = db.Table('PMultiLinearRegression_Tested_Fly', metadata, autoload=True, autoload_with=engine)


def SaveAsCSV():

    df_test = pd.read_sql_table( MultiLinearRegression_Test, con=engine)
    filepath_test = Path('/home/raha/Raha/Thesis/Data/PMultiLinearRegression_Tested_Fly.csv')  
    df_test.to_csv(filepath_test)

SaveAsCSV()