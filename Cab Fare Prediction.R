rm(list=ls())
setwd("E:/Study Material/Data Science/Learning Data Science edWisor/Projects/Cab fare prediction")
getwd()
#install.packages(c("ggplot2", "corrgram", "DMwR","DataCombine"))
train = read.csv("train_cab.csv", header = T, na.strings = c(" ", "", "NA"))
test = read.csv("test.csv")
test_pickup_datetime = test["pickup_datetime"]
str(train)


train$fare_amount = as.numeric(as.character(train$fare_amount))

#removing fare_amount which are less than 1 
train[which(train$fare_amount < 1 ),]
nrow(train[which(train$fare_amount < 1 ),])
train = train[-which(train$fare_amount < 1 ),]

View(train)


######Passenger_count variable
for (i in seq(4,11,by=1)){
  print(paste('passenger_count above ' ,i,nrow(train[which(train$passenger_count > i ),])))
}

train = train[-which(train$passenger_count < 1 ),]
train = train[-which(train$passenger_count > 6),]

dim(train)


summary(train$pickup_longitude)


#Latitudes range from -90 to 90.Longitudes range from -180 to 180.Removing which does not satisfy these ranges
print(paste('pickup_longitude above 180=',nrow(train[which(train$pickup_longitude >180 ),])))
print(paste('pickup_longitude above -180=',nrow(train[which(train$pickup_longitude < -180 ),])))
print(paste('pickup_latitude above 90=',nrow(train[which(train$pickup_latitude > 90 ),])))
print(paste('pickup_latitude above -90=',nrow(train[which(train$pickup_latitude < -90 ),])))
print(paste('dropoff_longitude above 180=',nrow(train[which(train$dropoff_longitude > 180 ),])))
print(paste('dropoff_longitude above -180=',nrow(train[which(train$dropoff_longitude < -180 ),])))
print(paste('dropoff_latitude above -90=',nrow(train[which(train$dropoff_latitude < -90 ),])))
print(paste('dropoff_latitude above 90=',nrow(train[which(train$dropoff_latitude > 90 ),])))


#### Removing irrelevant data which is beyond valid limits  
train = train[-which(train$pickup_latitude > 90),]

######################Missing value analysis######

#(train[which(train$pickup_longitude == 0 ),])
#(train[which(train$pickup_latitude == 0 ),])
#(train[which(train$dropoff_longitude == 0 ),])
#(train[which(train$pickup_latitude == 0 ),])

nrow(train[which(train$pickup_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])
nrow(train[which(train$dropoff_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])


# there are values which are equal to 0. we will remove them.

train = train[-which(train$pickup_longitude == 0),]
train = train[-which(train$dropoff_longitude == 0),]



####################Missing Value Analysis######################
################################################################

missing_val = data.frame(apply(train,2,function(x){sum(is.na(x))}))
#View(missing_val)
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_percentage"
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(train)) * 100
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]
View(missing_val)

train$passenger_count
unique(train$passenger_count)


nrow(train[which(train$passenger_count==1.3 ),])
train=train[-which(train$passenger_count==1.3 ),]

str(train)
train[,'passenger_count'] = factor(train[,'passenger_count'], labels=(1:6))
test[,'passenger_count'] = factor(test[,'passenger_count'], labels=(1:6))



#Defining a function for mode calculation
getmode <- function(x) {
  uniqv <- unique(x)
  uniqv[which.max(tabulate(match(x, uniqv)))]
}

####Missing value analysis for passenger_count

#Mode results are biased towards passenger_count=1
#train$passenger_count[120]
#train$passenger_count[120]=NA
#getmode(train$passenger_count)

#KNN imputation
library(DMwR)
library(class)
library(naniar)
#train = knnImputation(train, k = 13)
#train$passenger_count

#Summary of Missing Value Imputation For Passenger_count:
  # Actual value = 1
  # Mode = 1
  # KNN = 1


####Missing value analysis for fare


#train$fare_amount[1000]
#train$fare_amount[1000]=NA


df=train
# Mean Method
#mean(train$fare_amount, na.rm = T)

#Median Method
#median(train$fare_amount, na.rm = T)


# kNN Imputation
train = knnImputation(train, k = 11)

#train$fare_amount[1000]

#Actual value=18.1
#Mean= 15.11762
#Median=8.5
#KNN=23.18


sum(is.na(train))

rm(missing_val)


####################Outlier Analysis######################3

library(ggplot2)

pl1 = ggplot(train,aes(x = (passenger_count),y = fare_amount))
pl1 + geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

# Replace all outliers with NA and impute
vals = train[,"fare_amount"] %in% boxplot.stats(train[,"fare_amount"])$out
train[which(vals),"fare_amount"] = NA


sum(is.na(train$fare_amount))

#Imputing with KNN
train = knnImputation(train,k=11)


#Checking the missing values
sum(is.na(train$fare_amount))
str(train)



################# Feature Engineering##########################
###############################################################

sum(is.na(train))

#install.packages("lubridate")
library(lubridate)  

train$pickup_date = as.Date(as.character(train$pickup_datetime))
train$pickup_weekday = as.factor(format(train$pickup_date,"%u"))
train$pickup_mnth = as.factor(format(train$pickup_date,"%m"))
train$pickup_yr = as.factor(format(train$pickup_date,"%Y"))

train$pickuptime=ymd_hms(train$pickup_datetime)
breaks=hour(hm("00:00","05:00","11:00","17:00","21:00","23:59"))
labels=c("Night","Morning","Afternoon","Evening","Night")
train$Time_of_Day=cut(x=hour(train$pickuptime),breaks=breaks,labels=labels,include.lowest = TRUE)

#Add same features to test set
test$pickup_date = as.Date(as.character(test$pickup_datetime))
test$pickup_weekday = as.factor(format(test$pickup_date,"%u"))# Monday = 1
test$pickup_mnth = as.factor(format(test$pickup_date,"%m"))
test$pickup_yr = as.factor(format(test$pickup_date,"%Y"))

test$pickuptime=ymd_hms(test$pickup_datetime)
breaks=hour(hm("00:00","05:00","11:00","17:00","21:00","23:59"))
labels=c("Night","Morning","Afternoon","Evening","Night")
test$Time_of_Day=cut(x=hour(test$pickuptime),breaks=breaks,labels=labels,include.lowest = TRUE)




yt=as.Date(train$pickup_datetime)
ytt=as.Date(test$pickup_datetime)
class(yt)
#install.packages("hydroTSM")
library(hydroTSM)
train$Season=time2season(yt, out.fmt = "seasons", type="default")
test$Season=time2season(ytt, out.fmt = "seasons", type="default")



sum(is.na(train))# there was 1 'na' in pickup_datetime which created na's in above feature engineered variables.
train = na.omit(train)       #we will remove that 1 NA

str(train)
#############Applying One Hot Encoding ###########

#install.packages("fastDummies")
library(fastDummies)
train=dummy_cols(train,select_columns = c("Time_of_Day","Season","pickup_yr","passenger_count"))
test=dummy_cols(test,select_columns = c("Time_of_Day","Season","pickup_yr","passenger_count"))

train = subset(train,select = -c(pickup_datetime,pickup_date,pickuptime,Time_of_Day,pickup_mnth,Season,pickup_yr,passenger_count))
test = subset(test,select = -c(pickup_datetime,pickup_date,pickuptime,Time_of_Day,pickup_mnth,Season,pickup_yr,passenger_count))


library(geosphere)


# HAVERSINE distance in metres     ####We will not use this as this is less accurate for smaller distances#####
#train$Haversinedist=distHaversine(cbind(train$pickup_longitude,train$pickup_latitude),cbind(train$dropoff_longitude,train$dropoff_latitude), r=6378137)



#VINCENTY(Geodesic) distance in metres
train$geodesic=distGeo(cbind(train$pickup_longitude,train$pickup_latitude),cbind(train$dropoff_longitude,train$dropoff_latitude), a=6378137, f=1/298.257223563)

# Using Vincenty formula to calculate distance for test dataset also
test$geodesic=distGeo(cbind(test$pickup_longitude,test$pickup_latitude),cbind(test$dropoff_longitude,test$dropoff_latitude), a=6378137, f=1/298.257223563)


#Another method to calculate Vincenty
#library(Imap)
#train$dist1=gdist(train$pickup_longitude, train$pickup_latitude,train$dropoff_longitude,train$dropoff_latitude, units="km", a=6378137.0, b=6356752.3142, verbose = FALSE)



# Removing the variables which were used to feature engineer distance
train = subset(train,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
test = subset(test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))



##############Feature Selection ################  

####For numeric variable#######


numeric_index = sapply(train,is.numeric) #selecting only numeric
#View(numeric_index)
numeric_data = train[,numeric_index]

cnames = colnames(numeric_data)
str(numeric_data)

library(corrgram)
#Correlation analysis for numeric variables
corrgram(train[,numeric_index],upper.panel=panel.pie, main = "Correlation Plot")

str(train)

#View(cor(train[,numeric_index]))
#col=colorRampPalette(c("blue","white","red"))
#heatmap(x=numeric_data,col=col,symm=TRUE)
colnames(train)

####For categorical variable#########


train = subset(train,select = -c(passenger_count_1))
test = subset(test,select = -c(passenger_count_1,Time_of_Day_Night,Season_winter))

aov_results = aov(fare_amount ~ pickup_weekday +Time_of_Day_Morning+ Time_of_Day_Evening +Time_of_Day_Afternoon+ Season_spring + Season_autumm+Season_summer+passenger_count_2 +passenger_count_3+passenger_count_4+ passenger_count_5+passenger_count_6,data = train)

summary(aov_results)

0# Removing those variables whose p value is greater than 0.05 
train = subset(train,select=-c(pickup_weekday+Time_of_Day_Afternoon+Season_spring+Season_summer+passenger_count_2+passenger_count_3+passenger_count_4))

#remove from test set
test = subset(test,select=-c(pickup_weekday+Time_of_Day_Afternoon+Season_spring+Season_summer+passenger_count_2+passenger_count_3+passenger_count_4))

###############Feature Scaling################
#Normality check########

qqnorm(train$fare_amount)
histogram(train$fare_amount)
#install.packages(c("MASS","carDATA"))               # MASS package for truehist and carDATA for qqPlot
library(car)
library(MASS)
par(mfrow=c(1,2))
qqPlot(train$fare_amount)                             
truehist(train$fare_amount)                           
lines(density(train$fare_amount)) 

qqPlot(train$vincentydist)                             
truehist(train$vincentydist)                           
lines(density(train$vincentydist)) 
  
# qqPlot, it has a x values derived from gaussian distribution, if data is distributed normally then the sorted data points should lie very close to the solid reference line.
# truehist() scales the counts to give an estimate of the probability density.

# Normalizing Distance 

train[,'geodesic'] = (train[,'geodesic'] - min(train[,'geodesic']))/(max(train[,'geodesic'] - min(train[,'geodesic'])))

test[,'geodesic'] = (test[,'geodesic'] - min(test[,'geodesic']))/(max(test[,'geodesic'] - min(test[,'geodesic'])))


# #check multicollearity
#install.packages("usdm")
library(usdm)

#########Variance Inflation Factor#############
#vif(train[,-1])
#vifcor(train, th = 0.9)

View(numeric_data)



#train=train[,c(2,3,4,5,6,7,8,9,10,11,12,13,14,1)]
#test=test[,c(2,3,4,5,6,7,8,9,10,11,12,13,14,1)]

#################### Splitting train into train and validation subsets ###################
library(DataCombine)
library(sampling)
library(caret)
set.seed(1000)
tr.idx = createDataPartition(train$fare_amount,p=0.75,list = FALSE) # 75% in training and 25% in Validation Datasets
train_data = train[tr.idx,]
test_data = train[-tr.idx,] 

dftrain=train_data
dftest=test_data


View(test_data)
###########################################Model Deployment############################## 


#############            Linear regression               #################
lm_model = lm(fare_amount ~.,data=train_data)

summary(lm_model)

plot(lm_model$fitted.values,rstandard(lm_model),main ="Residual plot",xlab = "Predicted values of fare_amount",ylab = "standardized residuals")

dim(train_data)
lm_predictions = predict(lm_model,test_data[,2:23])

regr.eval(test_data[,1],lm_predictions, stats=c('mae','rmse','mape','mse'))

#################################Decision Tree############################

str(train)
library(rpart)
Dt_model = rpart(fare_amount ~ ., data = train_data, method = "anova")

summary(Dt_model)
#Predict for new test cases
predictions_DT = predict(Dt_model, test_data[,2:23])

qplot(x = test_data[,1], y = predictions_DT, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],predictions_DT)



################################Random Forest###########################

library(randomForest)
rf_model = randomForest(fare_amount ~.,data=train_data)

summary(rf_model)

rf_predictions = predict(rf_model,test_data[,2:23])

qplot(x = test_data[,1], y = rf_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],rf_predictions)

#########################XGBoost #######
#install.packages("xgboost")
library(xgboost)
train_data_matrix = as.matrix(sapply(train_data[-1],as.numeric))
test_data_data_matrix = as.matrix(sapply(test_data[-1],as.numeric))

xgboost_model = xgboost(data = train_data_matrix,label = train_data$fare_amount,nrounds = 15,verbose = FALSE)

summary(xgboost_model)
xgb_predictions = predict(xgboost_model,test_data_data_matrix)

regr.eval(test_data[,1],xgb_predictions)
