library(caret)
library(C50)
library(MASS)
library(randomForest)

trainData = read.csv("pml-training.csv")
featuresSuperSet = c('roll_belt', 'pitch_belt', 'yaw_belt', 'total_accel_belt', 
 'gyros_belt_x', 'gyros_belt_y', 'gyros_belt_z', 
 'accel_belt_x', 'accel_belt_y', 'accel_belt_z', 
 'magnet_belt_x', 'magnet_belt_y', 'magnet_belt_z',
 'roll_arm', 'pitch_arm', 'yaw_arm', 'total_accel_arm', 
 'gyros_arm_x', 'gyros_arm_y', 'gyros_arm_z', 
 'accel_arm_x', 'accel_arm_y', 'accel_arm_z', 
 'magnet_arm_x', 'magnet_arm_y', 'magnet_arm_z', 
 'roll_dumbbell', 'pitch_dumbbell', 'yaw_dumbbell', 'total_accel_dumbbell', 
 'gyros_dumbbell_x', 'gyros_dumbbell_y', 'gyros_dumbbell_z',
 'accel_dumbbell_x', 'accel_dumbbell_y', 'accel_dumbbell_z', 
 'magnet_dumbbell_x', 'magnet_dumbbell_y', 'magnet_dumbbell_z',
 'roll_forearm', 'pitch_forearm', 'yaw_forearm', 'total_accel_forearm', 
 'gyros_forearm_x', 'gyros_forearm_y', 'gyros_forearm_z', 
 'accel_forearm_x', 'accel_forearm_y', 'accel_forearm_z', 
 'magnet_forearm_x', 'magnet_forearm_y', 'magnet_forearm_z', 'classe')

trainData = subset(trainData, select = featuresSuperSet)
summary(trainData)

# check for missing values
lapply(is.na(train)[1:52], function(x) sum(x)) > 0

# Split train data into a train set and a validation set.
dataPartition <- createDataPartition(y = trainData$classe, p = 0.6, list = FALSE)
trainingSet <- trainData[dataPartition, ]
validationSet <- trainData[-dataPartition, ]
#nrow(trainingSet)

validationPartition <- createDataPartition(y = validationSet$classe, p = 0.5, list=FALSE)
modelComparisonSet <- validationSet[validationPartition, ]
modelPredictionSet <- validationSet[-validationPartition, ]
#nrow(validationSet)

# Train different models and compare
set.seed(555)

# C5.0 Algorithm
c50Model <- C5.0(trainingSet[, 0:52], trainingSet[, 53])
c50ValidationPredicted <- predict(c50Model, modelComparisonSet[, 0:52], type="class")
confusionMatrix(data = c50ValidationPredicted, modelComparisonSet[, 53])

# LDA
ldaModel <- lda(classe ~ ., data = trainingSet)
ldaValidationPredicted <- predict(ldaModel, modelComparisonSet[, 0:52], type="class")
confusionMatrix(data = ldaValidationPredicted$class, modelComparisonSet[, 53])

# Random Forests
rfModel <- randomForest(classe ~ ., data=trainingSet)
rfValidationPredicted <- predict(rfModel, modelComparisonSet[, 0:52], type="class")
confusionMatrix(data = rfValidationPredicted, modelComparisonSet[, 53])

# Compare the different models

# Estimate the metrics for the selected model
rfTestPredicted <- predict(rfModel, modelPredictionSet[, 0:52], type="class")
confusionMatrix(data = rfTestPredicted, modelPredictionSet[, 53])

# Predict assignment's test data
#testData <- read.csv("pml-test.csv")
