import os
from pyspark.sql import SparkSession
from osgeo import gdal, ogr, osr
import rasterio
import rasterio.features
import rasterio.warp
import numpy as np
from configs import *

def project_roads(in_tiff, in_feat, out_feat):

    tif = gdal.Open(in_tiff)

    driver = ogr.GetDriverByName("ESRI Shapefile")
    datasource = driver.Open(in_feat)
    layer = datasource.GetLayer()

    sourceprj = layer.GetSpatialRef()
    targetprj = osr.SpatialReference(wkt = tif.GetProjection())
    transform = osr.CoordinateTransformation(sourceprj, 
                                         targetprj)

    ds = ogr.GetDriverByName("ESRI Shapefile").CreateDataSource(out_feat)
    outlayer = ds.CreateLayer('', targetprj, ogr.wkbLineString)
    outlayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))

    i = 0

    for feature in layer:
        transformed = feature.GetGeometryRef()
        transformed.Transform(transform)

        geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb())
        defn = outlayer.GetLayerDefn()
        feat = ogr.Feature(defn)
        feat.SetField('id', i)
        feat.SetGeometry(geom)
        outlayer.CreateFeature(feat)
        i += 1
        feat = None
    
    ds = None

if __name__ == '__main__':
    if os.path.isfile(new_road_data) == False:
        project_roads(ref_tif, road_data, new_road_data)
        print("Road Projection Complete")
    else:
        print("Roads Already Projected")