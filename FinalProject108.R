rm(list = ls())

installIfAbsentAndLoad <- function(neededVector) {
  for(thispackage in neededVector) {
    if( ! require(thispackage, character.only = T) )
    { install.packages(thispackage)}
    require(thispackage, character.only = T)
  }
}

### Load required packages

needed <- c("rpart","dplyr", "reshape2", "tree",  "rattle","randomForest", "rfUtilities","ggplot2","doMC","caret","gbm")  
installIfAbsentAndLoad(needed)


set.seed(108)
### read in the cleaned data, remove unnecessary columns
data <- read.csv("ted412v2.csv")
data <- data[,-c(2, 4:5, 7:16, 18)]
str(data)

### double-check high correlations between columns
cor.mat <- as.matrix(cor(data))
cor.mat.melt <- arrange(melt(cor.mat), -abs(value))
dplyr::filter(cor.mat.melt, abs(value) < 1.0 & abs(value) > .5)

### remove high correlation columns and scale data
data <- data[, -c(8:10)]
data <- scale(data)
nrow(data)

#######################
### PCA and K-means ###
#######################

### run pca to reduce the dimensions, then use k-means on the first 2 principle components so we can
### graph the results. this might be too many clusters, so we should find a better k
pca <- prcomp(data)
k.10 <- kmeans(pca$x[,c(1:2)], 10, nstart = 50)
summary(k.10)
k.10$cluster
plot(pca$x[, c(1:2)], col = k.10$cluster)
pc1 <- pca$x[,1]
pc2 <- pca$x[,2]
cor(pc1, data)
cor(pc2, data)
(pc.corr <- data.frame(rbind(cor(pc1, data),
                             cor(pc2, data))))
### find a good k based off of within-sum of square distances
### wss function runs k-means from k = 1 to k.max and grabs the 
k.max <- 15
wss <- sapply(1:k.max, 
              function(k){kmeans(pca$x[, c(1:2)], k, nstart=50,iter.max = 15 )$tot.withinss})
wss
plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")

### k = 6 seems like the best k to use, so re-run k-means
best <- kmeans(pca$x[, c(1:2)], 6, nstart = 50)
plot(pca$x[, c(1:2)], col = best$cluster)

### now let's try and predict views

data <- as.data.frame(data)
train = sample(1:nrow(data), nrow(data)/2)
test <- data[-train,]

################################################
### Classification (1 = high views, 0 = low) ###
################################################
set.seed(108)
views <- data$views
cl <- data
cl$views[views > median(data$views)] <- 1
cl$views[views <= median(data$views)] <- 0
cl.train <- cl[train,]
cl.test <- cl[-train,]
#model.tree <- rpart(views ~ ., data <- cl, method = 'class')

###########################
### Classification Tree ###
###########################

### create maximum tree
max.model<-rpart(views ~.,data=cl.train ,method="class",
                 parms=list(split="information"),control=rpart.control(usesurrogate=0,
                                                                       maxsurrogate=0,cp=0,minbucket=1,minsplit=2)) 
fancyRpartPlot(max.model, main="Maximal Decision Tree")
print(max.model$cptable)
plotcp(max.model)

### prune mmax model
xerr<-max.model$cptable[,"xerror"] 
minxerr<-which.min(xerr)
mincp<-max.model$cptable[minxerr,"CP"]
model.prune<-prune(max.model,cp=mincp)
model.prune$cptable
fancyRpartPlot(model.prune, main="Decision Tree With Minimum C.V. Error")
#plot(model.prune)
#asRules(model.prune)

### false positives would harm us more than false negatives, so we will weight them more
weight.model<-rpart(views~.,data= cl, method="class",
                    parms=list(split="information"),control=rpart.control(usesurrogate=0,
                                                                          maxsurrogate=0,cp=0,minbucket=1,minsplit=2,loss=matrix(c(0,5,1,0), byrow=TRUE, nrow=2)))
plotcp(weight.model)
weight.model.prune<-prune(weight.model,cp=mincp)
weight.model.prune.predict <- predict(weight.model.prune, newdata=cl, type="class")

#confusion matrix
tab <- table(cl$views, weight.model.prune.predict,dnn=c("Actual", "Predicted"))
print(paste("Pruned weighted model min cross validation error: ", min(weight.model.prune$cptable[, "xerror"])))
sum(tab[1,2], tab[2,1]) / sum(tab)


#######################
#### Regression Tree ##
#######################
rm(list = ls())
data <- read.csv("ted412v2.csv")
data <- data[,-c(2, 4:5, 7:16, 18)]
data <- data[, -c(8:10)]
data <- scale(data)
set.seed(108)
data <- as.data.frame(data)
train = sample(1:nrow(data), nrow(data)/2)


tree.data=tree(views~.,data,subset=train)
summary(tree.data)

cv.data=cv.tree(tree.data, K = 10)
which.min.dev <- which.min(cv.data$dev)



plot(cv.data$size,cv.data$dev,type='b',xlab= "Tree Size", ylab = "CV Error")
title(main = 'Optimal Tree Size')
abline(v= cv.data$size[which.min.dev], lty = 'dashed',col = 'red')
legend("topright", inset = 0.05, paste('Best tree size =', as.character(cv.data$size[which.min.dev])),text.col = 'red')

prune.data=prune.tree(tree.data,best=cv.data$size[which.min.dev])

plot(prune.data)
text(prune.data,pretty=0)

summary(prune.data)

yhat=predict(prune.data,newdata=data[-train,])
data.test=data[-train,"views"]
tree.mse <- mean((yhat-data.test)^2)
tree.mse

#bagging

bag.data <- randomForest(views~.,data= data[train,], mtry = 6, importance = TRUE)
bag.data
yhat.bag <- predict(bag.data,newdata=data[-train,])
bag.mse <- mean((yhat.bag-data.test)^2)
bag.mse

#####################
### Random Forest ###
#####################

### use mtry as p/3, running on the train set

rf <- randomForest(views ~ ., data = data[train,], mtry = 2, importance = TRUE)
rf
yhat.rf <- predict(rf,newdata=data[-train,])
rf.mse <- mean((yhat.rf-data.test)^2)
rf.mse

#importance(rf)[order(importance(rf)[,"%IncMSE"], decreasing=T),]


### display a chart of Variable Importance
varImpPlot(rf, main="Variable Importance")


rf.caret <-train(views ~., data = data[train,],
                             method='rf',
                             importance=TRUE)
rf.caret


yhat.caret <- predict(rf.caret, data[-train,])
mean((yhat.caret - data.test)^2)

#the best model produced by rf.caret has a higher MSE on the test data set.

###boosting

boost <- gbm(views~. ,data = data[train,], distribution = 'gaussian', 
             n.trees = 5000, 
             interaction.depth = 4)

summary(boost)

boost.pred <- predict(boost, data[-train,], n.trees=5000)
mean((boost.pred - data.test)^2)

ctr <- trainControl(method = "cv", number = 10)

boost.caret <- train(views~., data = data[train,],
                     method='bstTree',
                     trControl=ctr)
plot(boost.caret)
boost.caret

boost.caret.pred <- predict(boost.caret, data[-train,])
mean((boost.caret.pred - data.test)^2)

