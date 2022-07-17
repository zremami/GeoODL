from owslib.wms import WebMapService
wms = WebMapService('http://wms.jpl.nasa.gov/wms.cgi', version='1.1.1')
wms.identification.type
'OGC:WMS'

wms.identification.version
'1.1.1'

wms.identification.title
'JPL Global Imagery Service'
wms.identification.abstract
'WMS Server maintained by JPL, worldwide satellite imagery.'
list(wms.contents)
['global_mosaic', 'global_mosaic_base', 'us_landsat_wgs84', 'srtm_mag', 'daily_terra_721', 'daily_aqua_721', 'daily_terra_ndvi', 'daily_aqua_ndvi', 'daily_terra', 'daily_aqua', 'BMNG', 'modis', 'huemapped_srtm', 'srtmplus', 'worldwind_dem', 'us_ned', 'us_elevation', 'us_colordem']
wms['global_mosaic'].title
'WMS Global Mosaic, pan sharpened'
wms['global_mosaic'].queryable
0
wms['global_mosaic'].opaque
0
wms['global_mosaic'].boundingBox
wms['global_mosaic'].boundingBoxWGS84
(-180.0, -60.0, 180.0, 84.0)
wms['global_mosaic'].crsOptions
['EPSG:4326', 'AUTO:42003']
wms['global_mosaic'].styles
{'pseudo_bright': {'title': 'Pseudo-color image (Uses IR and Visual bands, 542 mapping), gamma 1.5'}, 'pseudo': {'title': '(default) Pseudo-color image, pan sharpened (Uses IR and Visual bands, 542 mapping), gamma 1.5'}, 'visual': {'title': 'Real-color image, pan sharpened (Uses the visual bands, 321 mapping), gamma 1.5'}, 'pseudo_low': {'title': 'Pseudo-color image, pan sharpened (Uses IR and Visual bands, 542 mapping)'}, 'visual_low': {'title': 'Real-color image, pan sharpened (Uses the visual bands, 321 mapping)'}, 'visual_bright': {'title': 'Real-color image (Uses the visual bands, 321 mapping), gamma 1.5'}}
[op.name for op in wms.operations]
['GetCapabilities', 'GetMap']
wms.getOperationByName('GetMap').methods
{'Get': {'url': 'http://wms.jpl.nasa.gov/wms.cgi?'}}
wms.getOperationByName('GetMap').formatOptions
['image/jpeg', 'image/png', 'image/geotiff', 'image/tiff']
img = wms.getmap(   layers=['global_mosaic'],
styles=['visual_bright'],
srs='EPSG:4326',
bbox=(-112, 36, -106, 41),
size=(300, 250),
format='image/jpeg',
transparent=True)
out = open('jpl_mosaic_visb.jpg', 'wb')
out.write(img.read())
out.close()