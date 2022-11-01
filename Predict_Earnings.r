library("data.table")
library("cluster")
library("stats")
library("fpc")
library("MASS")
library("dbscan")
library("randomForest")
library("jmuOutlier")
getwd()


Data.Original <- unique(na.omit(fread("~/Downloads/Earnings_Train2022-1.csv")))[sample(.N, 2000, replace = FALSE),]


Numerical <- c("GPA", "Number_Of_Professional_Connections", "Earnings", "Graduation_Year", "Number_Of_Credits", "Number_Of_Parking_Tickets")


Categorical <- c("Major")


k.Hclust <- 6


Data.Original[, Categorical] <- lapply(Data.Original[, .SD, .SDcols = Categorical], function(x) {as.factor(x)})


Data <- Data.Original[, .SD, .SDcols = c(Numerical, Categorical)]


Diss <- as.dist(daisy(Data, metric = "gower"))


NMDS <- as.data.table(isoMDS(Diss, trace = FALSE)$points)


plot(NMDS, pch = 19, main = "Dissimilarity")
text(NMDS, pos = 1, labels = row.names(Data), cex = 0.5)


PAM <- pamk(Diss, diss = TRUE)$pamobject


Hclust <- hclust(Diss, method = "ward.D2")


plot(Hclust, labels = FALSE)


Hclust <- cutree(Hclust, k = k.Hclust)


HDBSCAN <- hdbscan(Diss, minPts = 31)


plot(HDBSCAN, show_flat = TRUE)


Data <- cbind(Data, PAM = as.factor(PAM$clustering), HCLUST = as.factor(Hclust), HDBSCAN = as.factor(HDBSCAN$cluster))


PAM <- Data[eval(PAM$id.med),]


HDBSCAN <- cbind(Data, PROB = HDBSCAN$membership_prob)[HDBSCAN != 0, .SD[PROB == max(PROB),], by = .(HDBSCAN)][, PROB := NULL]


plot(NMDS, pch = 19, main = "PAM", col = Data[, PAM])
text(NMDS, pos = 1, labels = Data[, PAM], cex = 0.5)


plot(NMDS, pch = 19, main = "HCLUST", col = Data[, HCLUST])
text(NMDS, pos = 1, labels = Data[, HCLUST], cex = 0.5)


plot(NMDS[Data[, HDBSCAN] != 0,], pch = 19, main = "HDBSCAN", col = as.numeric(as.character(Data[HDBSCAN != 0, HDBSCAN])))
text(NMDS[Data[, HDBSCAN] != 0,], pos = 1, labels = Data[HDBSCAN != 0, HDBSCAN], cex = 0.5)


Sample <- sample(nrow(Data), floor(0.75 * nrow(Data)), replace = FALSE)


Train <- Data[Sample,]
Test <- Data[!Sample,]


RF.PAM <- randomForest(x = Train[, .SD, .SDcols = !c("HDBSCAN", "PAM", "HCLUST")], y = Train[, PAM], xtest = Test[, .SD, .SDcols = !c("HDBSCAN", "PAM", "HCLUST")], ytest = Test[, PAM], ntree = 1101, importance = TRUE, proximity = TRUE, keep.forest = FALSE)
RF.HCLUST <- randomForest(x = Train[, .SD, .SDcols = !c("HDBSCAN", "PAM", "HCLUST")], y = Train[, HCLUST], xtest = Test[, .SD, .SDcols = !c("HDBSCAN", "PAM", "HCLUST")], ytest = Test[, HCLUST], ntree = 1101, importance = TRUE, proximity = TRUE, keep.forest = FALSE)
RF.HDBSCAN <- randomForest(x = Train[, .SD, .SDcols = !c("HDBSCAN", "PAM", "HCLUST")], y = Train[, HDBSCAN], xtest = Test[, .SD, .SDcols = !c("HDBSCAN", "PAM", "HCLUST")], ytest = Test[, HDBSCAN], ntree = 1101, importance = TRUE, proximity = TRUE, keep.forest = FALSE)


Error <- data.table(ERROR = c("OOB", "TEST"), PAM = c(mean(Train[, PAM] != RF.PAM$predicted) * 100, mean(Test[, PAM] != RF.PAM$test$predicted) * 100), HCLUST = c(mean(Train[, HCLUST] != RF.HCLUST$predicted) * 100, mean(Test[, HCLUST] != RF.HCLUST$test$predicted) * 100), HDBSCAN = c(mean(Train[, HDBSCAN] != RF.HDBSCAN$predicted) * 100, mean(Test[, HDBSCAN] != RF.HDBSCAN$test$predicted) * 100))


plot(RF.PAM, main = "RF PAM Predictors Error")
MDSplot(RF.PAM, Train[, PAM], pch = 19, main = "RF PAM")


plot(RF.HCLUST, main = "RF HCLUST Predictors Error")
MDSplot(RF.HCLUST, Train[, HCLUST], pch = 19, main = "RF HCLUST")


plot(RF.HDBSCAN, main = "RF HDBSCAN Predictors Error")
MDSplot(RF.HDBSCAN, Train[, HDBSCAN], pch = 19, main = "RF HDBSCAN")








Data.Original <- na.omit(fread("~/Downloads/Earnings_Train2022-1.csv"))


RF.EARN <- randomForest(x = Data.Original[, .SD, .SDcols = !c("Earnings")], y = Data.Original[, Earnings], ntree = 2777, mtry = 4, importance = TRUE, proximity = TRUE, keep.forest = TRUE)


RF.EARN


Test <- fread("~/Downloads/Earnings_Test_Students-1.csv")


predictions <- predict(RF.EARN, newdata = Test)


Test[, Earnings := predictions]



Sub <- fread("~/Downloads/earning_submission.csv")


Sub[, Earnings := predictions]



write.csv(Sub, "~/Downloads//earning_submission.csv", row.names = FALSE)







