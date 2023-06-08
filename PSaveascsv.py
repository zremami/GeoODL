
# import urllib library
# importing module
from pandas import *
import pandas as pd
import sqlalchemy as db
from sqlalchemy.orm import Session
from sqlalchemy import select
from pathlib import Path  

engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
connection = engine.connect()
metadata = db.MetaData()
MultiLinearRegression_Test = db.Table('PMultiLinearRegression_TTFly2', metadata, autoload=True, autoload_with=engine)


def SaveAsCSV():
    #query = db.select([odls]) 
    
    #ResultProxy = connection.execute(query)
    #ResultSet = ResultProxy.fetchall()
    df_test = pd.read_sql_table( MultiLinearRegression_Test, con=engine)
    filepath_test = Path('/home/raha/Raha/Thesis/Data/PMultiLinearRegression_TTFly2.csv')  
    df_test.to_csv(filepath_test)

    #df_train = pd.read_sql_table( MultiLinearRegression_Train, con=engine)
    #filepath_train = Path('/home/raha/Raha/Thesis/Data/MultiLinearRegression_Train.csv')  
    #df_train.to_csv(filepath_train)

SaveAsCSV()