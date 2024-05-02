import os
from configs import *
from pyspark.sql import SparkSession
from pyspark import SparkConf
import pyspark.sql.functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import col
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.classification import NaiveBayes



# https://spark.apache.org/docs/latest/ml-classification-regression.html


conf1 = SparkConf()
conf1.set("spark.executer.memory", "5g")
conf1.set("spark.driver.memory", "5g")

spark = SparkSession.builder.appName("Buffer").config(conf=conf1).getOrCreate()

df = spark.read.csv(data_table, 
                    header = True)

for col_name in ["B1","B2","B3","B4","B5","B6","B7","B10"]:
    df = df.withColumn(col_name, col(col_name).cast('float'))



df = VectorAssembler(inputCols =["B1","B2","B3","B4","B5","B6","B7","B10"], 
                            outputCol= "Features").transform(df)

# get_lbl_value = F.udf(lambda x: loc_lbl_value(x), IntegerType())

# df = df.withColumn("Label", get_lbl_value("Label"))

df = df.select(['features', 'Label'])
df = df.withColumn('Label', col('Label').cast('int'))



training = df.filter(df.Label != -1)
unk_data = df.filter(df.Label == -1)

splits = training.randomSplit([0.8, 0.2])
train_df = splits[0]
test_df = splits[1]


print(train_df.count(), test_df.count())

# Logistic Regression
lr = LogisticRegression(featuresCol='features', 
                      labelCol='Label', 
                      family = "multinomial",
                      maxIter=10, 
                      regParam=0.3,
                      elasticNetParam=0.8)

lr_model = lr.fit(train_df)

lr_Summary = lr_model.summary
print("Area Under ROC: %f" % lr_Summary.areaUnderROC)
# print("r2: %f" % lr_Summary.r2)

lr_predictions = lr_model.transform(test_df)
# lr_predictions.select('Prediction', "Label", 'features').show(5)

[print("labe %d: %s" % (i, prec)) for i, prec in enumerate(lr_Summary.precisionByLabel)]

lr_preds_and_labels = lr_predictions.select(['Prediction', 'Label']).withColumn('Label', 
                                                                             col('Label').cast(FloatType())).orderBy('Prediction')
lr_metrics = MulticlassMetrics(lr_preds_and_labels.rdd.map(tuple))
print(lr_metrics.confusionMatrix().toArray())



#Random Forest
rf = RandomForestClassifier(featuresCol="features", labelCol="Label")
rfModel = rf.fit(train_df)
rf_predictions = rfModel.transform(test_df)

rf_evaluator = MulticlassClassificationEvaluator(labelCol="Label", predictionCol="prediction")
rf_accuracy = rf_evaluator.evaluate(rf_predictions)
print("Random Forest Accuracy is %s" %(rf_accuracy))

rf_preds_and_labels = rf_predictions.select(['prediction', 'Label']).withColumn('Label', 
                                                                             col('Label').cast(FloatType())).orderBy('prediction')
rf_metrics = MulticlassMetrics(rf_preds_and_labels.rdd.map(tuple))
print(rf_metrics.confusionMatrix().toArray())


#Naive Bayes

nb = NaiveBayes(smoothing=1.0, 
                modelType="multinomial", 
                featuresCol='features',
                labelCol="Label")
nb_model = nb.fit(train_df)
nb_predictions = nb_model.transform(test_df)

nb_evaluator = MulticlassClassificationEvaluator(labelCol="Label", predictionCol="prediction")
nb_accuracy = nb_evaluator.evaluate(nb_predictions)
print("Naive Bayes is %s" %(nb_accuracy))


nb_preds_and_labels = nb_predictions.select(['prediction', 'Label']).withColumn('Label', 
                                                                             col('Label').cast(FloatType())).orderBy('prediction')
nb_metrics = MulticlassMetrics(nb_preds_and_labels.rdd.map(tuple))
print(nb_metrics.confusionMatrix().toArray())