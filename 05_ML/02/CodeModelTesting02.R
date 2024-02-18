#######################################################################
## Code - Model Testing 02: Dr. Ruhollah Taghizadeh
#######################################################################

# load library
library(sp)
library(raster)
library(sf)
library(caret)
library(randomForest)
library (ggplot2)

##############################################################################
# section 1: Digital Soil Mapping
# import Remote Sensing data
covariates_RS <- stack(list.files("./cov/", pattern="\\.tif$", full.names = TRUE))
names(covariates_RS) 

# import raster layers of covarites (DEM)
covariates_DEM <- stack(list.files("./SAGA/", pattern="\\.sdat$", full.names = TRUE))
names(covariates_DEM)

# stack layer from remote sensing, and terrain data
covariates = stack(covariates_RS,resample(covariates_DEM,covariates_RS))
names(covariates)

# aggregate rasters
# covariates = aggregate(covariates, 9, fun=mean)

# import the point soil data from desktop
point <- read.csv ("soil.csv", header = TRUE)

# convert soil point data to spatial point data  
coordinates(point) <- ~ x + y

# set projection
proj4string(point) <- CRS("+init=epsg:32639")

# plot the point on the raster
plot(covariates$Nir, main = "Landsat Image + Sample Points")
plot(point, add =T, pch = 19)

# extract covariate values at each point of observation 
cov = extract(covariates, point, method='bilinear', df=TRUE)

# combining covariates and soil properties
cov_soil = cbind(cov[,-1], ec=point$ec)
  
# remove na values
cov_soil <- cov_soil[complete.cases(cov_soil),]

# split the data to training (80%) and testing (20%) sets
train_index <- createDataPartition(cov_soil$ec, p = 0.8, list = FALSE, times = 1)

# subset the datasets
df_train <- cov_soil[ train_index,]
df_test  <- cov_soil[-train_index,]

# inspect the two datasets
str(df_train)
str(df_test)

# fit random forest model on training set
rf_fit <- randomForest(ec ~ ., data = df_train)

# visualize the importance of random forest
varImpPlot(rf_fit) 

# apply the random forest model on testing data
ec_pred <- predict(rf_fit, df_test)  

# calculate correlation
cor_rf <- cor(df_test$ec, ec_pred)
cor_rf

# calculate RMSE
RMSE_rf <- sqrt(mean((df_test$ec - ec_pred)^2))
RMSE_rf

# apply the RF model on the stack layer
map_rf <- raster::predict(covariates, rf_fit)

# plot the final map
spplot(map_rf, main="EC map based on RF model")



##############################################################################
# section 2: Random Forest Using Caret Package
# define traincontrol 1: default
ctr_1 <- trainControl(method="cv")

# train RF_1
RF_1 <- train(ec ~ ., 
                      data=cov_soil, 
                      method="rf", 
                      metric="RMSE", 
                      trControl = ctr_1)
RF_1

# define traincontrol 2: random search
ctr_2 <- trainControl(method="cv", search = 'random')

# train RF_2
RF_2 <- train(ec ~ .,
                      data=cov_soil, 
                      method='ranger', 
                      metric='RMSE', 
                      tuneLength = 5, 
                      trControl=ctr_2)
RF_2

# define traincontrol 3: grid search
ctr_3 <- trainControl(method="cv", search = 'grid')

# define tubegrids
tune_grid <- expand.grid(.mtry = (1:7)) 

# train RF_3
RF_3 <- train(ec ~ .,
                      data=cov_soil, 
                      method='rf', 
                      metric='RMSE', 
                      tuneGrid = tune_grid, 
                      trControl=ctr_3)
plot(RF_3)



##############################################################################
# section 3: Practice
# compute NDVI (nir - red / nir + red), add to the covariates, and plot it
# Is there any relationships between NDVI and EC map?
# compute the residuals for test data and plot the histograms
# train another model from (https://topepo.github.io/caret/available-models.html)





