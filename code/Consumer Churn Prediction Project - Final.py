#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from pyspark.sql import SparkSession
import pyspark.sql.functions as fn
import plotly.express as px
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import StringIndexer,VectorAssembler,QuantileDiscretizer
from functools import reduce
from pyspark.sql.window import *
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler,BucketedRandomProjectionLSH
from pyspark.sql.window import Window
from pyspark.ml.linalg import VectorUDT,SparseVector,Vectors
from pyspark.ml.clustering import KMeans
from pyspark.ml.classification import DecisionTreeClassifier,RandomForestClassifier
import pyspark.ml.evaluation as ev
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


# In[2]:


spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()
df = spark.read.csv("Telco_customer_churn.csv", header=True, inferSchema=True)


# ### EDA

# 1. Class Distribution

# In[3]:


# Group data by 'Churn Label' and count the number of distinct 'CustomerID' values in each group
grouped_data = df.groupby('Churn Value').agg(fn.countDistinct('CustomerID').alias('CustomerID'))

# Compute the total number of customers across all groups
total_customers = grouped_data.selectExpr('sum(CustomerID)').collect()[0][0]

# Compute the percentage of customers in each group relative to the total number of customers
grouped_data = grouped_data.withColumn('Percentage', fn.col('CustomerID') / total_customers * 100)

# Use Plotly Express to create a pie chart
fig = px.pie(grouped_data.toPandas(), 
             values='CustomerID', 
             names='Churn Value',
             hover_data={'Percentage':':.2f'})

fig.show()


# 2. Customer Distribution by City (Top 50)

# In[4]:


# count the number of customers of each city, order by total number desc
grouped_data = df.groupby('City').agg(fn.count('CustomerID').alias('NumOfCustomer'))
grouped_data = grouped_data.orderBy(fn.col('NumOfCustomer').desc()).limit(50)


fig = px.bar(grouped_data.toPandas(),
             x='City',
             y='NumOfCustomer',
             color='NumOfCustomer',
             text='NumOfCustomer')

fig.show()


# 3. Churn Rate of Previous Cities

# In[5]:


# Calculate churn rate by city
churn_data = df.groupBy('City').agg(fn.sum('Churn Value').alias('Churned'), fn.countDistinct('CustomerID').alias('TotalCustomers'))
churn_data = churn_data.withColumn('ChurnRate', churn_data.Churned / churn_data.TotalCustomers)
churn_data = churn_data.withColumn('ChurnRate', fn.round(churn_data.ChurnRate, 2)) # round to 2 decimal places

# Sort by descending total number of customers
churn_data = churn_data.sort(fn.col('TotalCustomers').desc())

# Limit to top 50 cities by total number of customers
churn_data = churn_data.limit(50)

# Plot line chart
fig = px.line(churn_data.toPandas(),
              x='City',
              y='ChurnRate',
              text='ChurnRate',
              labels={'City': 'City', 'ChurnRate': 'Churn Rate'})

fig.show()


# 4. Number of Customers and Churn Rate of Each Longitude and Latitude

# In[6]:


# sum the number of customers of each long and lat, calculate churn rate
grouped_data = df.groupby(['Latitude', 'Longitude']).agg(fn.countDistinct('CustomerID').alias('NumOfCustomer'), fn.sum(fn.col('Churn Value')).alias('ChurnValue'))
grouped_data = grouped_data.withColumn('ChurnRate', fn.col('ChurnValue')/fn.col('NumOfCustomer'))

# add churn rate and number of customers as value hovering on map
fig = px.scatter_mapbox(grouped_data.toPandas(), 
                        lat="Latitude", 
                        lon="Longitude", 
                        hover_data=['NumOfCustomer', 'ChurnRate'], 
                        zoom = 4, 
                        height=300)

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# 5. Distribution of Total Charges / Monthly Charges for Churn

# In[9]:


# round Total Charges and Monthly Charges to integers
charge_data = df.withColumn('Total Charges', fn.round(df['Total Charges']).cast('integer'))

# Create a boxplot to visualize the relationship between 'Churn Value' and 'Total Charges'
fig = px.box(charge_data.toPandas(), x='Churn Value', y='Total Charges')
fig.show()

# Create a boxplot to visualize the relationship between 'Churn Value' and 'Monthly Charges'
fig = px.box(charge_data.toPandas(), x='Churn Value', y='Monthly Charges')
fig.show()


# ### Data Preprocessing

# In[10]:


# see if there is any blank or missing values
df.select([fn.count(fn.when((fn.col(c) == ' ') | (fn.col(c).isNull()), c)).alias(c) for c in df.columns]).show()


# In[11]:


# https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset for column descriptions
# There is only one value for country, state, and count (US,CA,1) so we can drop
# CustomerID and Churn Reason we don't need for prediction
# Churn Label is repetitive, we have Churn Value already. Lat Long is also repetitive
df = df.drop("CustomerID","Count", "Churn Reason","Churn Label", "Country", "Lat Long", "State")


# In[12]:


# Churn reason is already dropped so no missing value other than Total Charges
# change the blank spaces in Total Charges to None so we can drop easily
df = df.withColumn('Total Charges', fn.when(df['Total Charges'] == ' ', None).otherwise(df['Total Charges']))
df = df.withColumn('Total Charges', df['Total Charges'].cast('double'))
print('Number of customers before dropping: {0}'.format(df.count()))
df = df.dropDuplicates()
df = df.na.drop()
print('Number of customers after dropping: {0}'.format(df.count()))


# In[13]:


df.dtypes


# In[14]:


# get the columns with string data type
string_cols = [c[0] for c in df.dtypes if c[1] == 'string']

string_indexers = [StringIndexer(inputCol=col, outputCol=col + '_index') for col in string_cols]

# fit the StringIndexers to the data
for indexer in string_indexers:
    df = indexer.fit(df).transform(df)

# drop the original columns that were indexed
df = df.drop(*string_cols)

# rename the indexed columns to their original names
for col_name in string_cols:
    df = df.withColumnRenamed(col_name + '_index', col_name)

# show the resulting DataFrame
df.show()


# In[15]:


# use VectorAssembler to turn all the feature columns into a vector
feature_cols = [c for c in df.columns if c != "Churn Value"]
featuresCreator = VectorAssembler(
inputCols=feature_cols,
outputCol='features'
)

df_vector = featuresCreator.transform(df).select("features", "Churn Value")
df_vector.show()


# In[16]:


# Calculate the Pearson correlation matrix
correlation_matrix = Correlation.corr(df_vector, "features").head()[0]

# Convert the correlation matrix to a Pandas DataFrame for easier display
correlation_matrix_df = pd.DataFrame(correlation_matrix.toArray(), index=feature_cols, columns=feature_cols)

# Set up the figure size and style
plt.figure(figsize=(15, 13))
sns.set(style="white")

# Generate the heatmap
sns.heatmap(correlation_matrix_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)

# Set up the title and axis labels
plt.title("Heatmap of Correlation Matrix", fontsize=20)
plt.xlabel("Features", fontsize=15)
plt.ylabel("Features", fontsize=15)
# Display the heatmap
plt.show()


# In[17]:


# Drop columns with correlation rate > 0.7 (drop one of them)
# we save zip code bc it contains info about Latitude and Longtitude and City
# save monthly charges, contains info about internet service
# save total charges, contains info about tenure months
# save streaming movies, contains info about streaming TV
# save online security, contains info about "Online Backup","Device Protection","Tech Support".
df = df.drop("City","Latitude","Longitude","Tenure Months","Internet Service","Streaming TV","Online Backup","Device Protection","Tech Support")


# In[18]:


feature_cols = [c for c in df.columns if c != "Churn Value"]
featuresCreator = VectorAssembler(
inputCols=feature_cols,
outputCol='features'
)

df_vector = featuresCreator.transform(df).select("features", "Churn Value")

# Calculate the Pearson correlation matrix
correlation_matrix = Correlation.corr(df_vector, "features").head()[0]

# Convert the correlation matrix to a Pandas DataFrame for easier display
correlation_matrix_df = pd.DataFrame(correlation_matrix.toArray(), index=feature_cols, columns=feature_cols)

# Set up the figure size and style
plt.figure(figsize=(15, 13))
sns.set(style="white")

# Generate the heatmap
sns.heatmap(correlation_matrix_df, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)

# Set up the title and axis labels
plt.title("Heatmap of Correlation Matrix", fontsize=20)
plt.xlabel("Features", fontsize=15)
plt.ylabel("Features", fontsize=15)
# Display the heatmap
plt.show()


# Discretize continuous features

# In[19]:


cont_cols = ["Zip Code", "Monthly Charges","Total Charges","Churn Score","CLTV"]
discretized_cols = [col + '_dis' for col in cont_cols]

# Create a QuantileDiscretizer object with input columns with 4 buckets, separating data 25% quantiles
discretizer = QuantileDiscretizer(numBuckets=4, inputCols=cont_cols, outputCols=discretized_cols)


# Fit and transform the data
df = discretizer.fit(df).transform(df)

# drop the original columns that were discretized
df = df.drop(*cont_cols)

# rename the indexed columns to their original names
for col_name in cont_cols:
    df = df.withColumnRenamed(col_name + '_dis', col_name)

# show the resulting DataFrame
df.show()


# In[20]:


# recreate the vectors
feature_cols = [c for c in df.columns if c != "Churn Value"]
featuresCreator = VectorAssembler(
inputCols=feature_cols,
outputCol='features'
)

df_vector = featuresCreator.transform(df).select("features", "Churn Value")


# ### Split training and testing data

# In[21]:


train_df, test_df = df_vector.randomSplit([0.7, 0.3])


# ### Data Resampling Methods

# In[22]:


# https://medium.com/@junwan01/oversampling-and-undersampling-with-pyspark-5dbc25cdf253 
major_df = train_df.filter(fn.col("Churn Value") == 0) # non-churn is the majority
minor_df = train_df.filter(fn.col("Churn Value") == 1)
ratio = round(major_df.count()/minor_df.count())
a = range(ratio)
print("ratio: {}".format(ratio))


# In[23]:


# oversampling

# duplicate the minority rows
oversampled_df = minor_df.withColumn("dummy", fn.explode(fn.array([fn.lit(x) for x in a]))).drop('dummy')
# combine both oversampled minority rows and previous majority rows
df_ros = major_df.unionAll(oversampled_df)
df_ros.show()
df_ros.groupBy("Churn Value").count().show()


# In[24]:


# Undersampling
sampled_majority_df = major_df.sample(False, 1/ratio)
df_rus = sampled_majority_df.unionAll(minor_df)
df_rus.show()
df_rus.groupBy("Churn Value").count().show()


# In[25]:


# https://gist.github.com/hwang018/420e288021e9bdacd133076600a9ea8c
def smote(vectorized_sdf):
    '''
    contains logic to perform smote oversampling, given a spark df with 2 classes
    inputs:
    * vectorized_sdf: cat cols are already stringindexed, num cols are assembled into 'features' vector
    output:
    * oversampled_df: spark df after smote oversampling
    '''
    dataInput_min = vectorized_sdf[vectorized_sdf['Churn Value'] == 1]
    dataInput_maj = vectorized_sdf[vectorized_sdf['Churn Value'] == 0]
    
    # LSH, bucketed random projection
    brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes",seed = 42, bucketLength = 100)
    # smote only applies on existing minority instances    
    model = brp.fit(dataInput_min)
    model.transform(dataInput_min)

    # here distance is calculated from brp's param inputCol
    self_join_w_distance = model.approxSimilarityJoin(dataInput_min, dataInput_min, float("inf"), distCol="EuclideanDistance")

    # remove self-comparison (distance 0)
    self_join_w_distance = self_join_w_distance.filter(self_join_w_distance.EuclideanDistance > 0)

    over_original_rows = Window.partitionBy("datasetA").orderBy("EuclideanDistance")

    self_similarity_df = self_join_w_distance.withColumn("r_num", fn.row_number().over(over_original_rows))

    self_similarity_df_selected = self_similarity_df.filter(self_similarity_df.r_num <= 4)

    over_original_rows_no_order = Window.partitionBy('datasetA')

    # list to store batches of synthetic data
    res = []
    
    @fn.udf(returnType=VectorUDT())
    def subtract_vector_udf(arr):
        # Must decorate func as udf to ensure that its callback form is the arg to df iterator construct
        a = arr[0]
        b = arr[1]
        if isinstance(a, SparseVector):
            a = a.toArray()
        if isinstance(b, SparseVector):
            b = b.toArray()
        array_ = a - b
        return random.uniform(0, 1) * Vectors.dense(array_)

    @fn.udf(returnType=VectorUDT())
    def add_vector_udf(arr):
        # Must decorate func as udf to ensure that its callback form is the arg to df iterator construct
        a = arr[0]
        b = arr[1]
        if isinstance(a, SparseVector):
            a = a.toArray()
        if isinstance(b, SparseVector):
            b = b.toArray()
        array_ = a + b
        return Vectors.dense(array_)
    
    # retain original columns
    original_cols = dataInput_min.columns
    
    for i in range(2): # how many batches of minority samples to generate
        print("generating batch %s of synthetic instances"%i)
        # logic to randomly select neighbour: pick the largest random number generated row as the neighbour
        df_random_sel = self_similarity_df_selected.withColumn("rand", fn.rand()).withColumn('max_rand', fn.max('rand').over(over_original_rows_no_order))                            .where(fn.col('rand') == fn.col('max_rand')).drop(*['max_rand','rand','r_num'])
        # create synthetic feature numerical part
        df_vec_diff = df_random_sel.select('*', subtract_vector_udf(fn.array('datasetA.features', 'datasetB.features')).alias('vec_diff'))
        df_vec_modified = df_vec_diff.select('*', add_vector_udf(fn.array('datasetA.features', 'vec_diff')).alias('features'))
        
        # for categorical cols, either pick original or the neighbour's cat values
        for c in original_cols:
            # randomly select neighbour or original data
            col_sub = random.choice(['datasetA','datasetB'])
            val = "{0}.{1}".format(col_sub,c)
            if c != 'features':
                # do not unpack original numerical features
                df_vec_modified = df_vec_modified.withColumn(c,fn.col(val))
        
        # this df_vec_modified is the synthetic minority instances,
        df_vec_modified = df_vec_modified.drop(*['datasetA','datasetB','vec_diff','EuclideanDistance'])
        
        res.append(df_vec_modified)
    
    dfunion = reduce(DataFrame.unionAll, res)
    # union synthetic instances with original full (both minority and majority) df
    oversampled_df = dfunion.union(vectorized_sdf.select(dfunion.columns))
    
    return oversampled_df


# In[26]:


# SMOTE
df_smote = smote(train_df)
df_smote.show()
df_smote.groupBy("Churn Value").count().show()


# ### Models

# In[27]:


# train the k-means model
kmeans = KMeans(k=2)


# In[28]:


# k-means + original
label_df = train_df.select("features")
ori_model = kmeans.fit(label_df)

# make predictions on the test data
ori_predictions = ori_model.transform(test_df)

# show the predictions
ori_predictions.show()


# In[29]:


# k-means + ROS
label_df = df_ros.select("features")
ros_model = kmeans.fit(label_df)

# make predictions on the test data
ros_predictions = ros_model.transform(test_df)

# show the predictions
ros_predictions.show()


# In[30]:


# k-means + RUS
label_df = df_rus.select("features")
rus_model = kmeans.fit(label_df)

# make predictions on the test data
rus_predictions = rus_model.transform(test_df)

# show the predictions
rus_predictions.show()


# In[31]:


# k-means + SMOTE
label_df = df_smote.select("features")
smote_model = kmeans.fit(label_df)

# make predictions on the input data
smote_predictions = smote_model.transform(test_df)

# show the predictions
smote_predictions.show()


# In[32]:


# Creating a Decision Tree model
dt = DecisionTreeClassifier(labelCol="Churn Value", featuresCol="features")


# In[33]:


# DT + original
# Fitting the model on the training data
dt_model = dt.fit(train_df)

# Making predictions on the test data
dt_ori_predictions = dt_model.transform(test_df)
dt_ori_predictions.show()


# In[34]:


# DT + ROS
# Fitting the model on the training data
dt_model = dt.fit(df_ros)

# Making predictions on the test data
dt_ros_predictions = dt_model.transform(test_df)
dt_ros_predictions.show()


# In[35]:


## DT + RUS
# Fitting the model on the training data
dt_model = dt.fit(df_rus)

# Making predictions on the test data
dt_rus_predictions = dt_model.transform(test_df)
dt_rus_predictions.show()


# In[36]:


# DT + SMOTE
# Fitting the model on the training data
dt_model = dt.fit(df_smote)

# Making predictions on the test data
dt_smote_predictions = dt_model.transform(test_df)
dt_smote_predictions.show()


# In[37]:


# Create an instance of the RandomForestClassifier
rf = RandomForestClassifier(labelCol='Churn Value', featuresCol="features", seed=42)


# In[38]:


# RF + original
# Train the model
model = rf.fit(train_df)

# Make predictions on the test data
rf_ori_predictions = model.transform(test_df)
rf_ori_predictions.show()


# In[39]:


# RF + ROS
# Train the model
model = rf.fit(df_ros)

# Make predictions on the test data
rf_ros_predictions = model.transform(test_df)
rf_ros_predictions.show()


# In[40]:


# RF + RUS
# Train the model
model = rf.fit(df_rus)

# Make predictions on the test data
rf_rus_predictions = model.transform(test_df)
rf_rus_predictions.show()


# In[41]:


# RF + SMOTE
# Train the model
model = rf.fit(df_smote)

# Make predictions on the test data
rf_smote_predictions = model.transform(test_df)
rf_smote_predictions.show()


# ### Model Evaluations

# In[42]:


AUC_evaluator = ev.BinaryClassificationEvaluator(rawPredictionCol='prediction',labelCol='Churn Value')
Acc_evaluator = ev.MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="Churn Value", metricName="accuracy")
F1_evaluator = ev.MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="Churn Value", metricName="f1")


# In[43]:


# K-means + original
ori_predictions = ori_predictions.withColumn("prediction", fn.col("prediction").cast("double"))
print("areaUnderROC", AUC_evaluator.evaluate(ori_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(ori_predictions))
print("F1-score", F1_evaluator.evaluate(ori_predictions))


# In[44]:


# K-means + ROS
ros_predictions = ros_predictions.withColumn("prediction", fn.col("prediction").cast("double"))
print("areaUnderROC", AUC_evaluator.evaluate(ros_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(ros_predictions))
print("F1-score", F1_evaluator.evaluate(ros_predictions))


# In[45]:


# K-means + RUS
rus_predictions = rus_predictions.withColumn("prediction", fn.col("prediction").cast("double"))
print("areaUnderROC", AUC_evaluator.evaluate(rus_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(rus_predictions))
print("F1-score", F1_evaluator.evaluate(rus_predictions))


# In[46]:


# K-means + SMOTE
smote_predictions = smote_predictions.withColumn("prediction", fn.col("prediction").cast("double"))
print("areaUnderROC", AUC_evaluator.evaluate(smote_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(smote_predictions))
print("F1-score", F1_evaluator.evaluate(smote_predictions))


# In[47]:


# DT + original
print("areaUnderROC", AUC_evaluator.evaluate(dt_ori_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(dt_ori_predictions))
print("F1-score", F1_evaluator.evaluate(dt_ori_predictions))


# In[48]:


# DT + ROS
print("areaUnderROC", AUC_evaluator.evaluate(dt_ros_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(dt_ros_predictions))
print("F1-score", F1_evaluator.evaluate(dt_ros_predictions))


# In[49]:


# DT + RUS
print("areaUnderROC", AUC_evaluator.evaluate(dt_rus_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(dt_rus_predictions))
print("F1-score", F1_evaluator.evaluate(dt_rus_predictions))


# In[50]:


# DT + SMOTE
print("areaUnderROC", AUC_evaluator.evaluate(dt_smote_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(dt_smote_predictions))
print("F1-score", F1_evaluator.evaluate(dt_smote_predictions))


# In[51]:


# RF + original
print("areaUnderROC", AUC_evaluator.evaluate(rf_ori_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(rf_ori_predictions))
print("F1-score", F1_evaluator.evaluate(rf_ori_predictions))


# In[52]:


# RF + ROS
print("areaUnderROC", AUC_evaluator.evaluate(rf_ros_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(rf_ros_predictions))
print("F1-score", F1_evaluator.evaluate(rf_ros_predictions))


# In[53]:


# RF + RUS
print("areaUnderROC", AUC_evaluator.evaluate(rf_rus_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(rf_rus_predictions))
print("F1-score", F1_evaluator.evaluate(rf_rus_predictions))


# In[54]:


# RF + SMOTE
print("areaUnderROC", AUC_evaluator.evaluate(rf_smote_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(rf_smote_predictions))
print("F1-score", F1_evaluator.evaluate(rf_smote_predictions))


# ### Hyperparameter Tuning

# cannot perform cross-validation with unlabeled data in k-means clustering

# In[55]:


paramGrid = ParamGridBuilder()     .addGrid(dt.maxDepth, [2, 5, 10])     .addGrid(dt.maxBins, [10, 20, 30])     .build()

# Define the cross-validation method
cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=F1_evaluator, numFolds=3)

# Fit the cross-validation model to the training data
cvModel = cv.fit(train_df)

# Get the best model
bestModel = cvModel.bestModel

bestMaxDepth = bestModel.getMaxDepth()
bestMaxBins = bestModel.getMaxBins()
print(bestMaxDepth)
print(bestMaxDepth)


# In[56]:


dt_best_predictions = bestModel.transform(test_df)
print("areaUnderROC", AUC_evaluator.evaluate(dt_best_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(dt_best_predictions))
print("F1-score", F1_evaluator.evaluate(dt_best_predictions))


# areaUnderROC: 0.8516907851190961
# Accuracy score: 0.8927550047664442
# F1-score: 0.8910510116355512

# In[57]:


paramGrid = (ParamGridBuilder()
             .addGrid(rf.numTrees, [10, 20, 30])
             .addGrid(rf.maxDepth, [2, 5, 10])
             .build())

# Define the cross-validation method
cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=F1_evaluator, numFolds=3)

# Fit the cross-validation model to the training data
cvModel = cv.fit(train_df)

# Get the best model
bestModel = cvModel.bestModel
bestModel


# In[58]:


rf_best_predictions = bestModel.transform(test_df)
print("areaUnderROC", AUC_evaluator.evaluate(rf_best_predictions,
{AUC_evaluator.metricName: 'areaUnderROC'}))
print("Accuracy score", Acc_evaluator.evaluate(rf_best_predictions))
print("F1-score", F1_evaluator.evaluate(rf_best_predictions))


# areaUnderROC:0.8783519195451014
# Accuracy score:0.8976268031642625
# F1-score:0.8984842224472432
