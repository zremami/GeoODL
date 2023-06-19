# this class was created to extract the location geometry of the selected station to show in QGIS
from pandas import *
def pStationLocation():
    import pandas as pd
    import numpy as np
    import sqlalchemy as db
    from pandas_geojson import to_geojson
    import json  


    #connect to database
    engine = db.create_engine('postgresql://postgres:123456@localhost:5432/geoODLdb')
    connection = engine.connect()
    metadata = db.MetaData()
    odls_precipitations = db.Table('odls_precipitations_dummy', metadata, autoload=True, autoload_with=engine)
    df =pd.read_sql_table( odls_precipitations, con=engine)

    # get CSV
    sondenstandorte = read_csv("/home/raha/Raha/Thesis/Data/odl_sondenstandorte.csv")
    df_sondenstandorte= pd.DataFrame(sondenstandorte)
    df_sondenstandorte = df_sondenstandorte.set_index(df_sondenstandorte['locality_code'])

    #group by locality code
    df_gropby= df.groupby('Locality_code').mean()


    # join 
    df3 = df_gropby.join(df_sondenstandorte, lsuffix='_odl_prep', rsuffix='df_sondenstandorte', how='inner')

    #select properties
    df3 = df3[['locality_code','locality_name','longitude','latitude']]

    #convert to json
    geo_json = to_geojson(df3, lat='latitude', lon='longitude',
                 properties=['locality_code','locality_name'])

    # convert to geojson
    from pandas_geojson import write_geojson
    write_geojson(geo_json, filename='pstationLocation.geojson', indent=4)

    with open('pstationLocation' + '.json', 'w', encoding='utf-8') as f:
      json.dump(geo_json, f, ensure_ascii=False, indent=4)

pStationLocation()
