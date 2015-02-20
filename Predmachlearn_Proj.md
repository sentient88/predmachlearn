# Practical Machine Learning Course Project

<br/>
## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit, it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

In this project, we will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. Data source: <http://groupware.les.inf.puc-rio.br/har>

The goal of this project is to predict the manner in which they did the exercise (refer to "classe" variable in the data set). After building the prediction model, it will be used to predict 20 different test cases which are provided as part of the assignment.

<br/>
## Data Preparation & Processing

Download the "training" and "testing" data sets to the working directory. We note that the data sets are in the form of comma-separated-value (csv) files.

We first read in the **training** data from the raw csv file "pml-training.csv": 


```r
raw_data <- read.csv("pml-training.csv", na.strings= c("NA","","#DIV/0!"))
summary(raw_data)
```

As there are many variables with overly high percentage of NA values in the data set, it would be best to exclude them from the prediction model. Hence, we proceed to remove the columns that contain NA values from the data set. The first 7 columns (denoting record IDs, user IDs, timestamps and other non-measurements) are removed too.


```r
NA_columns <- apply(raw_data, 2, function(x) {sum(is.na(x))})
clean_data <- raw_data[, which(NA_columns == 0)]
library(dplyr)
clean_data <- select(clean_data, -(X:num_window))
```

Using the clean data set "clean_data", we split it into training set (60%) and testing set (40%) by "classe" variable. 

This will allow us to perform cross-validation of the prediction model later (i.e. to build the model using the training set and then evaluate using the testing set which has not been used in the training of model).


```r
library(caret)
set.seed(133)
inTrain <- createDataPartition(y = clean_data$classe, p = 0.6, list = FALSE)
training <- clean_data[inTrain, ]
testing <- clean_data[-inTrain, ]
dim(training); dim(testing)
```

```
## [1] 11776    53
```

```
## [1] 7846   53
```

<br/>
## Exploratory Data Analysis

Next, we check for "near zero variance" predictors. If found, they should be removed from the training set.


```r
nzv <- nearZeroVar(training, saveMetrics=TRUE)
nzv
```

```
##                      freqRatio percentUnique zeroVar   nzv
## roll_belt             1.077064    8.82302989   FALSE FALSE
## pitch_belt            1.088496   13.62941576   FALSE FALSE
## yaw_belt              1.137124   14.48709239   FALSE FALSE
## total_accel_belt      1.053550    0.22927989   FALSE FALSE
## gyros_belt_x          1.039333    1.02751359   FALSE FALSE
## gyros_belt_y          1.131253    0.54347826   FALSE FALSE
## gyros_belt_z          1.072106    1.39266304   FALSE FALSE
## accel_belt_x          1.014523    1.31623641   FALSE FALSE
## accel_belt_y          1.113687    1.15489130   FALSE FALSE
## accel_belt_z          1.138462    2.37771739   FALSE FALSE
## magnet_belt_x         1.037037    2.55604620   FALSE FALSE
## magnet_belt_y         1.142473    2.36073370   FALSE FALSE
## magnet_belt_z         1.003571    3.63451087   FALSE FALSE
## roll_arm             51.375000   19.33593750   FALSE FALSE
## pitch_arm            76.148148   22.64775815   FALSE FALSE
## yaw_arm              33.145161   21.50985054   FALSE FALSE
## total_accel_arm       1.027778    0.55197011   FALSE FALSE
## gyros_arm_x           1.038462    5.26494565   FALSE FALSE
## gyros_arm_y           1.425325    3.08254076   FALSE FALSE
## gyros_arm_z           1.085366    1.93614130   FALSE FALSE
## accel_arm_x           1.046729    6.45380435   FALSE FALSE
## accel_arm_y           1.080292    4.44972826   FALSE FALSE
## accel_arm_z           1.128205    6.39436141   FALSE FALSE
## magnet_arm_x          1.000000   11.15828804   FALSE FALSE
## magnet_arm_y          1.018519    7.20108696   FALSE FALSE
## magnet_arm_z          1.046875   10.55536685   FALSE FALSE
## roll_dumbbell         1.189873   87.88213315   FALSE FALSE
## pitch_dumbbell        1.904255   85.91202446   FALSE FALSE
## yaw_dumbbell          1.305556   87.32167120   FALSE FALSE
## total_accel_dumbbell  1.068460    0.35665761   FALSE FALSE
## gyros_dumbbell_x      1.044444    1.95312500   FALSE FALSE
## gyros_dumbbell_y      1.373529    2.25883152   FALSE FALSE
## gyros_dumbbell_z      1.019284    1.58797554   FALSE FALSE
## accel_dumbbell_x      1.029703    3.44769022   FALSE FALSE
## accel_dumbbell_y      1.000000    3.87228261   FALSE FALSE
## accel_dumbbell_z      1.097902    3.38824728   FALSE FALSE
## magnet_dumbbell_x     1.019231    8.92493207   FALSE FALSE
## magnet_dumbbell_y     1.279279    6.90387228   FALSE FALSE
## magnet_dumbbell_z     1.008850    5.51970109   FALSE FALSE
## roll_forearm         13.271186   14.75883152   FALSE FALSE
## pitch_forearm        71.181818   21.24660326   FALSE FALSE
## yaw_forearm          14.493827   14.30027174   FALSE FALSE
## total_accel_forearm   1.123016    0.57744565   FALSE FALSE
## gyros_forearm_x       1.118750    2.33525815   FALSE FALSE
## gyros_forearm_y       1.038793    6.00373641   FALSE FALSE
## gyros_forearm_z       1.100000    2.33525815   FALSE FALSE
## accel_forearm_x       1.018182    6.56419837   FALSE FALSE
## accel_forearm_y       1.096774    8.29653533   FALSE FALSE
## accel_forearm_z       1.020408    4.66202446   FALSE FALSE
## magnet_forearm_x      1.060000   12.13485054   FALSE FALSE
## magnet_forearm_y      1.150943   15.24286685   FALSE FALSE
## magnet_forearm_z      1.000000   13.44259511   FALSE FALSE
## classe                1.469065    0.04245924   FALSE FALSE
```

Hence, we note that there is no variable with near zero variance.

<br/>
## Training The Prediction Model

We choose "Random Forests" method to fit the model because of its well-known high level of accuracy among current algorithms. Moreover, it runs efficiently on large data bases and is able to handle thousands of input variables.



```r
library(randomForest)
set.seed(7841)
modelFit <- randomForest(classe ~ ., data = training)
modelFit
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.64%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3345    3    0    0    0 0.0008960573
## B   12 2262    5    0    0 0.0074594120
## C    0   14 2039    1    0 0.0073028238
## D    1    0   27 1899    3 0.0160621762
## E    0    0    5    4 2156 0.0041570439
```

<br/>
In Sample Error:

We note that the prediction model has a very small OOB estimate of error rate which is 0.64%. 

**Hence, we would expect the Out of Sample Error to be greater than 0.64%.** 

The Out of Sample Error can be estimated with cross-validation approach which is detailed below. 

<br/>
## Cross Validation

To evaluate the prediction model: 

We apply the model to the testing set to get the predictions. The predictions are then compared with the actual reference values in the testing set.


```r
predictions <- predict(modelFit, newdata = testing)
confusionMatrix(predictions, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2230    9    0    0    0
##          B    1 1509    6    0    0
##          C    0    0 1362   18    0
##          D    0    0    0 1266    3
##          E    1    0    0    2 1439
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9931, 0.9964)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9941   0.9956   0.9844   0.9979
## Specificity            0.9984   0.9989   0.9972   0.9995   0.9995
## Pos Pred Value         0.9960   0.9954   0.9870   0.9976   0.9979
## Neg Pred Value         0.9996   0.9986   0.9991   0.9970   0.9995
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1923   0.1736   0.1614   0.1834
## Detection Prevalence   0.2854   0.1932   0.1759   0.1617   0.1838
## Balanced Accuracy      0.9988   0.9965   0.9964   0.9920   0.9987
```

We note that this prediction model gives a high level of accuracy (0.9949). 

Hence, we can confidently proceed to predict the 20 different test cases provided in the file "pml-testing.csv".

<br/>
## Predictions for 20 Test Cases

We read in the **testing** data from the raw csv file "pml-testing.csv": 


```r
testcases <- read.csv("pml-testing.csv", na.strings= c("NA","","#DIV/0!"))
```

We then apply the model to the data set "testcases" to get the predictions. 


```r
predict_testcases <- predict(modelFit, newdata = testcases)
```

The predictions are submitted in appropriate format to the programming assignment for automated grading.

Results: The predictions for all 20 test cases are correct! 

<br/>
## The End

<br/>

```r
## Submission for 20 test cases:
## 
## pml_write_files = function(x){
##  n = length(x)
##  for(i in 1:n){
##    filename = paste0("problem_id_",i,".txt")
##    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
##   }
##  }
## 
## pml_write_files(predict_testcases)
## This will create 20 text files in the working directory for submission.
```
