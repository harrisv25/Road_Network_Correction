import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, create_map, lit
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import IntegerType
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

ref = gdal.Open('data/input/image/LC09_L2SP_124051_20240407_20240408_02_T1_SR_B1.TIF')

geo_t = ref.GetGeoTransform()
x_size= ref.RasterXSize
y_size = ref.RasterYSize


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
roads = np.array(ds.GetRasterBand(1).ReadAsArray())

rds_idx = np.where(roads == 255)

data = list(zip(rds_idx[0].astype(int), rds_idx[1]))

# data = data[:10000]

spark = SparkSession.builder.appName("Buffer").getOrCreate()
df = spark.createDataFrame([([int(d[0]) - 1, int(d[0]), int(d[0]) + 1], 
                             [int(d[1]) - 1, int(d[1]), int(d[1]) + 1]) for d in data], 
                           ["OSM_RDS_IDX_X", "OSM_RDS_IDX_Y"])


df = df.select(explode(df.OSM_RDS_IDX_Y).alias("OSM_RDS_IDX_Y"), df.OSM_RDS_IDX_X)
df = df.select(explode(df.OSM_RDS_IDX_X).alias("OSM_RDS_IDX_X"), df.OSM_RDS_IDX_Y)


df = df.where((df.OSM_RDS_IDX_X > -1) & (df.OSM_RDS_IDX_Y > -1) & (df.OSM_RDS_IDX_X < roads.shape[0]) & (df.OSM_RDS_IDX_Y < roads.shape[1]))
df = df.dropDuplicates(["OSM_RDS_IDX_X", "OSM_RDS_IDX_Y"])

df = df.withColumn("OSM_RDS_IDX_X", df["OSM_RDS_IDX_X"].cast(IntegerType()))
df = df.withColumn("OSM_RDS_IDX_Y", df["OSM_RDS_IDX_Y"].cast(IntegerType()))

def loc_img_value(x):
    return int(array[int(x[0])][int(x[1])])

for b in [i for i in os.listdir('data/input/image') if "B" in i]:
    b_label = b.split('_')[-1].split('.')[0]
    band = gdal.Open('data/input/image/{}'.format(b))
    array = np.array(band.GetRasterBand(1).ReadAsArray())

    df = VectorAssembler(inputCols =["OSM_RDS_IDX_X", "OSM_RDS_IDX_Y"], 
                          outputCol= "{}".format(b_label)).transform(df)
    
    get_img_value = F.udf(lambda x: loc_img_value(x), IntegerType())

    df = df.withColumn("{}".format(b_label), get_img_value("{}".format(b_label)))

    if b_label != "B10":
        df = df.withColumn("{}".format(b_label), 
                      ((col("{}".format(b_label)) * .0000275) + -0.2))
        df = df.withColumn("{}".format(b_label), 
                      F.when(df["{}".format(b_label)] < 0, 0).otherwise(df["{}".format(b_label)]))
    else:
        df = df.withColumn("{}".format(b_label), 
                      ((col("{}".format(b_label)) * .00341802) + 149))

df.describe().show()
# vector =  VectorAssembler(inputCols =["OSM_RDS_IDX_X", "OSM_RDS_IDX_Y"], 
#                           outputCol= "b1").transform(df)
# def loc_img_value(x):
#     return int(array[int(x[0])][int(x[1])])

# get_img_value = F.udf(lambda x: loc_img_value(x), IntegerType())

# vector = vector.withColumn("B1", get_img_value("b1"))

# # vector.describe().show()


