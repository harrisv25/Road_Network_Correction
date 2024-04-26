import os
from pyspark.sql import SparkSession
from pyspark.sql.types import ShortType
from osgeo import gdal, ogr, osr
import rasterio
import rasterio.features
import rasterio.warp
import numpy as np
# import geopandas as gpd

# https://spark.apache.org/docs/latest/ml-classification-regression.html

# test = gpd.read_file('./data/input/roads/Study_Area_Roads.shp')

# img = rasterio.open('data/input/image/LC09_L2SP_124051_20240407_20240408_02_T1_QA_PIXEL.TIF')
# print(img.count, img.height, img.width, img.crs, img.dtypes, img.bounds)

tif = gdal.Open('data/input/image/LC09_L2SP_124051_20240407_20240408_02_T1_QA_PIXEL.TIF')

geo_t = tif.GetGeoTransform()
x_size= tif.RasterXSize
y_size = tif.RasterYSize


xmin = min(geo_t[0], geo_t[0] + x_size * geo_t[1])
xmax = max(geo_t[0], geo_t[0] + x_size * geo_t[1])
ymin = min(geo_t[3], geo_t[3] + y_size * geo_t[5])
ymax = max(geo_t[3], geo_t[3] + y_size * geo_t[5])


ds = gdal.Rasterize('./data/input/roads/Study_Area_Roads.tif', './data/input/roads/Study_Area_Roads_prj.shp', 
                    xRes=30, 
                    yRes=30,
                    format="GTIFF",
                    allTouched=True, 
                    outputBounds=[xmin, ymin, xmax, ymax], 
                    noData = -1,
                    outputType=gdal.GDT_Int16)


# #road_values should be 255, else -1
# roads = np.array(ds.GetRasterBand(1).ReadAsArray())

# rds_idx = np.where(roads == 255)

# data = list(zip(rds_idx[0].astype(int), rds_idx[1]))

# spark = SparkSession.builder.appName("Buffer").getOrCreate()
# df = spark.createDataFrame([(int(d[0]), int(d[1])) for d in data], 
#                            ["OSM_RDS_IDX_X", "OSM_RDS_IDX_Y"])

# df.show()

# array = np.array(tif.GetRasterBand(1).ReadAsArray())
