import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode, col, create_map, lit
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import VectorUDT
from pyspark.sql.types import IntegerType
from osgeo import gdal, ogr, osr
import numpy as np
import pandas as pd
from pyspark.ml.regression import LinearRegression

# https://spark.apache.org/docs/latest/ml-classification-regression.html


ref = gdal.Open('data/input/image/LC09_L2SP_124051_20240407_20240408_02_T1_SR_B1.TIF')

geo_t = ref.GetGeoTransform()
x_size= ref.RasterXSize
y_size = ref.RasterYSize


xmin = min(geo_t[0], geo_t[0] + x_size * geo_t[1])
xmax = max(geo_t[0], geo_t[0] + x_size * geo_t[1])
ymin = min(geo_t[3], geo_t[3] + y_size * geo_t[5])
ymax = max(geo_t[3], geo_t[3] + y_size * geo_t[5])


ds = gdal.Rasterize('./data/input/roads/Study_Area_Roads.tif', './data/input/roads/Study_Area_Roads_prj/Study_Area_Roads_prj.shp', 
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


data = list(zip(rds_idx[0].astype(int), rds_idx[1].astype(int)))


# data = data[:100000]

def chunker(seq, size):
     return (seq[pos:pos+size] for pos in range(0, len(seq), size))

spark = SparkSession.builder.appName("Buffer").getOrCreate()
for chunk in chunker(data, 10000):
    if 'df' not in locals():
        df = spark.createDataFrame([([int(d[0]) - 1, int(d[0]), int(d[0]) + 1], 
                                 [int(d[1]) - 1, int(d[1]), int(d[1]) + 1]) for d in chunk], 
                               ["OSM_RDS_IDX_X", "OSM_RDS_IDX_Y"])

        df = df.select(explode(df.OSM_RDS_IDX_Y).alias("OSM_RDS_IDX_Y"), df.OSM_RDS_IDX_X)
        df = df.select(explode(df.OSM_RDS_IDX_X).alias("OSM_RDS_IDX_X"), df.OSM_RDS_IDX_Y)


        df = df.where((df.OSM_RDS_IDX_X > -1) & 
                      (df.OSM_RDS_IDX_Y > -1) & 
                      (df.OSM_RDS_IDX_X < roads.shape[0]) & 
                      (df.OSM_RDS_IDX_Y < roads.shape[1]))
        df = df.dropDuplicates(["OSM_RDS_IDX_X", "OSM_RDS_IDX_Y"])

        df = df.withColumn("OSM_RDS_IDX_X", df["OSM_RDS_IDX_X"].cast(IntegerType()))
        df = df.withColumn("OSM_RDS_IDX_Y", df["OSM_RDS_IDX_Y"].cast(IntegerType()))
        # print(df.count())
    elif 'df' in locals():
        temp = spark.createDataFrame([([int(d[0]) - 1, int(d[0]), int(d[0]) + 1], 
                                 [int(d[1]) - 1, int(d[1]), int(d[1]) + 1]) for d in chunk], 
                               ["OSM_RDS_IDX_X", "OSM_RDS_IDX_Y"])

        temp = temp.select(explode(temp.OSM_RDS_IDX_Y).alias("OSM_RDS_IDX_Y"), temp.OSM_RDS_IDX_X)
        temp = temp.select(explode(temp.OSM_RDS_IDX_X).alias("OSM_RDS_IDX_X"), temp.OSM_RDS_IDX_Y)


        temp = temp.where((temp.OSM_RDS_IDX_X > -1) & 
                  (temp.OSM_RDS_IDX_Y > -1) & 
                  (temp.OSM_RDS_IDX_X < roads.shape[0]) & 
                  (temp.OSM_RDS_IDX_Y < roads.shape[1]))
        temp = temp.dropDuplicates(["OSM_RDS_IDX_X", "OSM_RDS_IDX_Y"])

        temp = temp.withColumn("OSM_RDS_IDX_X", temp["OSM_RDS_IDX_X"].cast(IntegerType()))
        temp = temp.withColumn("OSM_RDS_IDX_Y", temp["OSM_RDS_IDX_Y"].cast(IntegerType()))

        df = df.union(temp)
        # print(df.count())
# For Development purposes. Create a visual raster for training data selection
# m1= np.array(df.select("OSM_RDS_IDX_X").collect()).ravel()
# m2= np.array(df.select("OSM_RDS_IDX_Y").collect()).ravel()

# sel_roads = np.zeros(roads.shape, dtype=int)

# sel_roads[m1, m2] = 1

# driver = gdal.GetDriverByName('GTIFF')
# new = driver.Create('./data/input/roads/Sel_Roads.tif',
    #                      ds.RasterXSize,
    #                      ds.RasterYSize,
    #                      1, 
    #                      gdal.GDT_Int16
    #                      )

# new.GetRasterBand(1).WriteArray(sel_roads)
# new.GetRasterBand(1).SetNoDataValue(-1)
# new.SetGeoTransform(geo_t)
# new.SetProjection(ds.GetProjection())


dsr = gdal.Rasterize('./data/input/Training_Data.tif', './data/input/Training_Data/Training_Data.shp', 
                    xRes=30, 
                    yRes=30,
                    format="GTIFF",
                    allTouched=True, 
                    attribute = "Road",
                    outputBounds=[xmin, ymin, xmax, ymax], 
                    noData = -1,
                    outputType=gdal.GDT_Int16)


# #road_values should be 255, else -1
labels = np.array(dsr.GetRasterBand(1).ReadAsArray())


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



def loc_lbl_value(x):
        return int(labels[int(x[0])][int(x[1])])

df = VectorAssembler(inputCols =["OSM_RDS_IDX_X", "OSM_RDS_IDX_Y"], 
                          outputCol= "Label").transform(df)

get_lbl_value = F.udf(lambda x: loc_lbl_value(x), IntegerType())

df = df.withColumn("Label", get_lbl_value("Label"))

df = df.where((df.B1 > 0) | 
              (df.B2 > 0) | 
              (df.B3 > 0) | 
              (df.B4 > 0) |
              (df.B5 > 0) |
              (df.B6 > 0) |
              (df.B7 > 0)
              )

