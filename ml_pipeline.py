import findspark
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
findspark.init()

from pyspark.sql import SparkSession
import os
import warnings

# Set the SPARK_LOCAL_IP environment variable if needed
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

# Suppress warnings
warnings.filterwarnings('ignore')

# Build the SparkSession with extra Java options
spark = SparkSession.builder \
    .appName("NASA Airfoil Self Noise ETL") \
    .config("spark.driver.extraJavaOptions", "--add-opens java.base/java.nio=ALL-UNNAMED --add-opens java.base/sun.nio.ch=ALL-UNNAMED") \
    .config("spark.executor.extraJavaOptions", "--add-opens java.base/java.nio=ALL-UNNAMED --add-opens java.base/sun.nio.ch=ALL-UNNAMED") \
    .config("spark.sql.legacy.allowUntypedScalaUDF", "true") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.sql.repl.eagerEval.enabled", "true") \
    .getOrCreate()


# Load the CSV file into a DataFrame
df = spark.read.csv("NASA_airfoil_noise_raw.csv", header=True, inferSchema=True)

# Print top 5 rows of the dataset
df.show(5)

# Print the total number of rows in the dataset
rowcount1 = df.count()
print(rowcount1)

# Drop all duplicate rows from the dataset
df = df.dropDuplicates()

# Print the total number of rows in the dataset after removing duplicates
rowcount2 = df.count()
print(rowcount2)

# Drop all rows that contain null values from the dataset
df = df.dropna()

# Print the total number of rows in the dataset after removing null values
rowcount3 = df.count()
print(rowcount3)

# Store the cleaned data in parquet format
df.write.parquet("cleaned_airfoil_noise_data.parquet")

df = df.withColumnRenamed("SoundLevel", "SoundLevelDecibels")
df.write.parquet("NASA_airfoil_noise_cleaned.parquet")

print("Part 1 - Evaluation")
print("Total rows = ", rowcount1)
print("Total rows after dropping duplicate rows = ", rowcount2)
print("Total rows after dropping duplicate rows and rows with null values = ", rowcount3)
print("New column name = ", df.columns[-1])

import os
print("NASA_airfoil_noise_cleaned.parquet exists :", os.path.isfile("NASA_airfoil_noise_cleaned.parquet"))

df = spark.read.parquet("NASA_airfoil_noise_cleaned.parquet")
rowcount4 = df.count()
print(rowcount4)

# Define the VectorAssembler pipeline stage
assembler = VectorAssembler(
    inputCols=[col for col in df.columns if col != "SoundLevelDecibels"],
    outputCol="features"
)

# Define the StandardScaler pipeline stage
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures"
)

# Define the Model creation pipeline stage
lr = LinearRegression(
    featuresCol="scaledFeatures",
    labelCol="SoundLevelDecibels"
)

# Build the pipeline
pipeline = Pipeline(stages=[assembler, scaler, lr])

# Split the data into training and testing sets with 70:30 split
(trainingData, testingData) = df.randomSplit([0.7, 0.3], seed=42)

# Fit the pipeline using the training data
pipelineModel = pipeline.fit(trainingData)

# Evaluation
print("Part 2 - Evaluation")
print("Total rows =", rowcount4)
ps = [str(x).split("_")[0] for x in pipeline.getStages()]
print("Pipeline Stage 1 =", ps[0])
print("Pipeline Stage 2 =", ps[1])
print("Pipeline Stage 3 =", ps[2])
print("Label column =", lr.getLabelCol())

predictions = pipelineModel.transform(testingData)

# Print the MSE
evaluator = RegressionEvaluator(
    labelCol="SoundLevelDecibels",
    predictionCol="prediction",
    metricName="mse"
)
mse = evaluator.evaluate(predictions)
print("Mean Squared Error =", mse)

# Print the MAE
evaluator.setMetricName("mae")
mae = evaluator.evaluate(predictions)
print("Mean Absolute Error =", mae)

# Print the R-Squared (R2)
evaluator.setMetricName("r2")
r2 = evaluator.evaluate(predictions)
print("R Squared =", r2)

# Evaluation
print("Part 3 - Evaluation")
print("Mean Squared Error =", round(mse, 2))
print("Mean Absolute Error =", round(mae, 2))
print("R Squared =", round(r2, 2))

lrModel = pipelineModel.stages[-1]
print("Intercept =", round(lrModel.intercept, 2))

# Save the model to the path "Final_Project"
pipelineModel.save("Final_Project")

# Load the model from the path "Final_Project"
from pyspark.ml import PipelineModel
loadedPipelineModel = PipelineModel.load("Final_Project")

# Make predictions using the loaded model on the test data
predictions = loadedPipelineModel.transform(testingData)

# Show the predictions
predictions.select("SoundLevelDecibels", "prediction").show(5)

# Evaluation
print("Part 4 - Evaluation")
loadedmodel = loadedPipelineModel.stages[-1]
totalstages = len(loadedPipelineModel.stages)
inputcolumns = loadedPipelineModel.stages[0].getInputCols()
print("Number of stages in the pipeline =", totalstages)
for i, j in zip(inputcolumns, loadedmodel.coefficients):
    print(f"Coefficient for {i} is {round(j, 4)}")

# Stop Spark Session
spark.stop()





