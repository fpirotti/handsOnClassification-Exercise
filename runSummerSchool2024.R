if (!require("pacman")) install.packages("pacman")
pacman::p_load(data.table, h2o,  devtools, emojifont)

if(!require(CloudGeometry)){ 
  devtools::install_github("fpirotti/CloudGeometry")
  library(CloudGeometry)
} 
 
# https://github.com/fpirotti/CloudGeometry

## read points from Vaihingen benchmark 400K points lidar 
## (http://www2.isprs.org/commissions/comm3/wg4/tests.html)
## 
## 
#path2script <- rstudioapi::getSourceEditorContext()$path
#setwd(dirname(path2script))
train<-fread("data/ISPRSbenchmark/Vaihingen3D_Training.pts")
test <-fread("data/ISPRSbenchmark/Vaihingen3D_EVAL_WITH_REF.pts")
## no column names!
names(train)
## re-assign column names 
names(train)<-c("x","y","z", "int","return","returns", "class")
names(test)<-c("x","y","z", "int","return","returns", "class")
## show classes and frequencies
train.classes = table(train$class)

## class names
cl<-c('Powerline','Low vegetation','Impervious surfaces','Car','Fence/Hedge','Roof','Facade','Shrub','Tree')
print(cl)
names(train.classes)<-cl
 
##  RDS file with 23 features, e.g. 23 columns with "geometric features" 
##  extracted from cloud compare with 1 m neighbour distance
#features<-readRDS("features.rds")
#head(features)
#
## if not into CloudCompare more features 
## from https://github.com/fpirotti/CloudGeometry
## TRAIN DATA FEATURES CALCULATION -------------------
features.train2a <- CloudGeometry::calcGF(train[,1:3])
features.train2b <- CloudGeometry::calcGF(train[,1:3],2)
features.train2c <- CloudGeometry::calcGF(train[,1:3],4, threads = 14)
features.train2 <- cbind(features.train2a, features.train2b, features.train2c)
## add my features (they are in the same order so we only need to bind horizontally)
## NB let's get rid of XYZ... it might learn BAD BAD things
##      (spatial autocorrelation....)
train.data<-cbind(train[,-c(1:3)], features.train2) 
## change class id (integer) to factor 
## class ID to class name
classes <- cl[ train.data$class+1 ]
train.data$class<- factor(classes, levels=cl )

train.classes <- table(train.data$class)
plot(table(train.data$class))
text(4.5, mean(train.classes), labels=paste("unbalanced! ", emoji('frowning')), cex=3.5, col='red',
     family='EmojiOne')

## TEST DATA FEATURES CALCULATION -------------------
features.test2a <- CloudGeometry::calcGF(test[,1:3])
features.test2b <- CloudGeometry::calcGF(test[,1:3],2, verbose =T)
features.test2c <- CloudGeometry::calcGF(test[,1:3],4, verbose = T, threads = 14)
features.test2 <- cbind(features.test2a, features.test2b, features.test2c)
## add my features (they are in the same order so we only need to bind horizontally)
test.data<-cbind(test[,-c(1:3)], features.test2)
## change class id (integer) to factor 
## class ID to class name
classes <- cl[ test.data$class+1 ]
test.data$class<- factor(classes, levels=cl)

### new with H2o --------

library(h2o)
h2o.init()
df <- h2o::as.h2o(train.data)
df.test <- h2o::as.h2o(test.data)


splits <- h2o.splitFrame(data = df,
                         ratios = c(0.5)  # 50/50 training/testing
                          )
splits.train <- splits[[1]]
splits.test <- splits[[2]]

## RANDOM FOREST AND DEEP LEARNING ----
### MODELLING -----
rf <- h2o.randomForest(y = "class",
                       training_frame = df,
                       model_id = "our.rf",
                      
                       seed = 1234)
h2o.saveModel(rf, path = "models/summerSchool2")
# rf <- h2o.loadModel(path = "models/summerSchool2/our.rf")

h2o.confusionMatrix(rf)
 
## PERFORMANCE ----
# rf_perf1 <- h2o.performance(model = rf, newdata = splits.test)
rf_perf2 <- h2o.performance(model = rf, newdata = df.test)
h2o.confusionMatrix(rf_perf2)
# dl <- h2o.deeplearning(y = "class",
#                        training_frame = splits.train,
#                        model_id = "our.dl",
#                        seed = 1234)
# dl_perf1 <- h2o.performance(model = dl, newdata = test)
  
## VARIABLE IMPORTANCE ----
h2o.varimp_plot(rf)
 
## PREDICTION ----
df.pred <- h2o.predict(rf, newdata = df.test)
df.pred.df<-as.data.frame(df.pred) 
head(df.pred.df)

df.pred.df.prob <- apply(df.pred.df[,-1], 1, function(x){round(max(x),3) })


final.df <- data.frame(test[,c("x", "y", "z" ,"class")], class.pred=as.integer(df.pred.df$predict), 
                       prob=df.pred.df.prob)


final.df$class.pred <- as.integer(final.df$class.pred)-1
write.csv(final.df,"classified.txt", row.names = F)
zip("classified.zip", "classified.txt")
