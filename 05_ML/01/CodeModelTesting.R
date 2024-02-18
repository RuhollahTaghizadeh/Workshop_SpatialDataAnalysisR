#######################################################################
## Code - Model Testing: Dr. Ruhollah Taghizadeh
#######################################################################

# Section 1 ###########################################################
# empty memory and workspace
rm(list=ls())

# check directory
getwd()

# to install all required R packages
ls <- c("sp", "raster", "sf", "mapview", "corrplot",
        "caret", "rpart", "rpart.plot", "randomForest")
new.packages <- ls[!(ls %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

# load library
library(sp)
library(raster)
library(sf)
library(mapview)
library(corrplot)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)

# Import covaraites (Remote Sensing, Terrain, Rainfall, NPP)
# import raster layers of covariates (Remote Sensing data)
blue = raster("./RS/Blue.tif")
green = raster("./RS/Green.tif")
red = raster("./RS/Red.tif")
nir = raster("./RS/Nir.tif")

# calculate some Remote Sensing indices such as NDVI
ndvi = (nir - red) / (nir + red)

# make a stack layer from Remote Sensing data
landsat = stack(blue,green,red,nir,ndvi)

# set the names to the rasters in the stack file
names(landsat) <- c("blue","green","red","nir","ndvi")

# inspect the structure of stack layer
landsat
   
# plot the stack layers of Remote Sensing data
plot(landsat)

# import the terrain data using list.files command
dem_lst <- list.files("./Terrain/", pattern="\\.sdat$", full.names = TRUE)

# inspect the structure of list
dem_lst

# make a stack layer from terrain data 
terrain <- stack(dem_lst)

# plot the stack layers of terrain data
plot(terrain)

# Re-sampling Remote Sensing data based on terrain data
RS_terrain = resample(landsat, terrain, method="bilinear")

# import mean annual rainfall (mm)
rain <- raster("./Climate/rainfall.sdat")

# Re-sampling rainfall data based on terrain data
rain_terrain = resample(rain, terrain, method="bilinear")

# import Terra Net Primary Production kg*C/m^2
NPP <- raster("./NPP/NPP.sdat")

# Re-sampling NPP based on terrain data
NPP_terrain = resample(NPP,terrain, method="bilinear")

# make a stack file from re-sampled remote sensing data, terrain data, rainfall, and NPP
covariates_stack = stack(terrain, RS_terrain, rain_terrain, NPP_terrain)

# inspect the structure of stack layer
names(covariates_stack)

# plot the stack of covariates
plot(covariates_stack)




# Section 2 ###########################################################
# Import point data shapefile
point = st_read("./Shapefile", "points")

# Import Bavaria boundary shapefile
Bavaria = st_read("./Shapefile", "Bavaria")


# plot the point on the raster
plot(covariates_stack$DEM, main = "DEM", xlab = "Easting (m)", ylab = "Northing (m)")
plot(point,add =T, pch = 19)
plot(Bavaria, add =T)

# plot an interactive map
mapview::mapview(point, zcol = "OC", at = c(0,5,10,15,20,25,30,35,200), legend = TRUE)

# inspect the point shape files
# types of the spatial data
class(point)

# summary statistics of data
summary(point)

# histogram of SOC
hist(point$OC,col ="blue", xlab= "SOC (g/kg)", main="Histogram")




# Section 3 ###########################################################
# extract covariate values at each point of observation 
cov = extract(covariates_stack, point, method='bilinear', df=TRUE)

# inspect the data.frame of cov
str(cov)

# combining covariates and soil properties
cov_soil = cbind(cov[,-1], OC=point$OC)
  
# inspect the data.frame of cov_soil
str(cov_soil)
  
# check the correlation covariates and OC
corrplot.mixed(cor(cov_soil), lower.col = "black", number.cex = .7)




# Section 4 ###########################################################
# remove na values
cov_soil <- cov_soil[complete.cases(cov_soil),]

# remove high values of oc
cov_soil <- cov_soil[cov_soil$OC<78,]

# check number of rows
nrow(cov_soil)

# check number of column 
ncol(cov_soil)

# split the data to training (80%) and testing (20%) sets
trainIndex <- createDataPartition(cov_soil$OC, 
                                  p = 0.8, list = FALSE, times = 1)

# subset the datasets
cov_soil_Train <- cov_soil[ trainIndex,]
cov_soil_Test  <- cov_soil[-trainIndex,]

# inspect the two datasets
dim(cov_soil_Train)
dim(cov_soil_Test)

# fit a linear regression on training set
linear_fit <- lm(OC ~ CNBL+DEM+MCA+Slope+
                   TWI+blue+green+red+nir+
                   ndvi+rainfall+NPP, 
                 data=cov_soil_Train)

# look at the summary of linear model
summary(linear_fit)
broom::tidy(linear_fit)
broom::glance(linear_fit)

# apply the linear model on testing data
OC_linear_Pred <- predict(linear_fit, cov_soil_Test)  

# check the plot actual and predicted OC values
plot(cov_soil_Test$OC, OC_linear_Pred, main="Linear model", 
     col="blue",xlab="Actual OC", ylab="Predicted OC", 
     xlim=c(0,100),ylim=c(0,100))
abline(coef = c(0,1),  col="red" )

# calculate correlation
cor_linear <- cor(cov_soil_Test$OC, OC_linear_Pred)
cor_linear

# calculate RMSE
RMSE_linear <- sqrt(mean((cov_soil_Test$OC - OC_linear_Pred)^2))
RMSE_linear




# Section 5 ###########################################################
# fit decision tree model on training set
tree_fit <- rpart(OC ~ CNBL+DEM+MCA+
                    Slope+TWI+blue+green+
                    red+nir+ndvi+rainfall+NPP,
                  method="anova", data=cov_soil_Train,
                  control = rpart.control(minsplit = 15, cp = 0.05))

# display the results of decision tree
printcp(tree_fit) 

# visualize cross-validation results
plotcp(tree_fit) 

# visualize the tree
rpart.plot(tree_fit) 

# apply the tree model on testing data
OC_tree_Pred <- predict(tree_fit, cov_soil_Test)  

# check the plot actual and predicted OC values
plot(cov_soil_Test$OC, OC_tree_Pred, main="Tree model", 
     col="blue",xlab="Actual OC", ylab="Predicted OC", xlim=c(0,100),ylim=c(0,100))
abline(coef = c(0,1),  col="red" )

# calculate correlation
cor_tree <- cor(cov_soil_Test$OC, OC_tree_Pred)
cor_tree

# calculate RMSE
RMSE_tree <- sqrt(mean((cov_soil_Test$OC - OC_tree_Pred)^2))
RMSE_tree




# Section 6 ###########################################################
# fit random forest model on training set
rf_fit <- randomForest(OC ~ CNBL+DEM+MCA+Slope+
                         TWI+blue+green+red+nir
                       +ndvi+rainfall+NPP, 
                       data=cov_soil_Train,
                       ntree=1000, do.trace = 25)

# visualize the importance of random forest
varImpPlot(rf_fit) 

# apply the random forest model on testing data
OC_rf_Pred <- predict(rf_fit, cov_soil_Test)  

# check the plot actual and predicted OC values
plot(cov_soil_Test$OC, OC_rf_Pred, main="Tree model", 
     col="blue",xlab="Actual OC", ylab="Predicted OC", xlim=c(0,100),ylim=c(0,100))
abline(coef = c(0,1),  col="red" )

# calculate correlation
cor_rf <- cor(cov_soil_Test$OC, OC_rf_Pred)
cor_rf

# calculate RMSE
RMSE_rf <- sqrt(mean((cov_soil_Test$OC - OC_rf_Pred)^2))
RMSE_rf




# Section 7 ###########################################################
# check the accuracy of three models
RMSE_models <- c(Linear=RMSE_linear,Tree=RMSE_tree,RF=RMSE_rf)
cor_models <- c(Linear=cor_linear,Tree=cor_tree,RF=cor_rf)

# plot the final results
dev.off()
par(mfrow=c(1,2))
barplot(RMSE_models, main="RMSE",col=c("red","blue","green"))
barplot(cor_models, main="Correlation",col=c("red","blue","green"))




# Section 8 ###########################################################
# apply the best model on the stack layer
map_rf <- raster::predict( covariates_stack,rf_fit)

# plot the final maps
spplot(map_rf, main="SOC map based on RF model")

# plot an interactive map
mapview::mapview(map_rf)

