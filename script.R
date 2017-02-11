library(ggplot2)
library(lattice)
library(caret)
library(xgboost)
library(plyr)

set.seed(125)

trainUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"


trainData <- read.csv(url(trainUrl), na.strings=c("NA","#DIV/0!",""))
testData <- read.csv(url(testUrl), na.strings=c("NA","#DIV/0!",""))

thresholdNas <- as.integer(.7*nrow(trainData))

##Pre process data
trainData$X <- NULL
trainData$cvtd_timestamp <- NULL
trainData <- trainData[, !colSums(is.na(trainData)) > thresholdNas]
testData <- testData[, which(names(testData) %in% names(trainData))]
trainLabels <- trainData$classe
trainData$classe <- NULL

##Convert data to be used on xgboost

trainMatrix <- data.matrix(trainData)
testMatrix <- data.matrix(testData)

## Model selection using CV

##Search Grid
xgb_grid = expand.grid(
  nrounds = 200,
  eta = c(0.3,0.1,0.05,0.03),  
  max_depth = c(4,6,8,10)
)

##Train Control
xgb_train_control <- trainControl(
  method = "cv",
  number = 10,
  verboseIter = FALSE,
  returnData = FALSE,
  returnResamp = "all",                                                      
  classProbs = TRUE,    
  allowParallel = TRUE
)

xgb_train <- train(
  x=trainMatrix,
  y=trainLabels,
  trControl = xgb_train_control,
  tuneGrid = xgb_grid,
  method = "xgbTree"
)

transformNumberIntoClass <- function(x) {
  if(x==1) 'A'
  else if(x==2) 'B'
  else if(x==3) 'C'
  else if(x==4) 'D'
  else if(x==5) 'E'
}

xgbFit <- xgb_train$finalModel

predictions <- predict(xgbFit, testMatrix)
predictionsMatrix <- matrix(predictions, nrow=nrow(testMatrix), ncol=5, byrow=T)
finalPredictions <- apply(predictionsMatrix, 1, function(x) which(x==max(x)))
dfPredictions <- data.frame(TestCase=1:20, 
                            Class= unlist(lapply(finalPredictions, transformNumberIntoClass)))