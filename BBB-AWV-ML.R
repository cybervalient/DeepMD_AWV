#Se hace lo mismo con las otras bases
base_BBB_ES<-read.csv("/media/ubuntu/YANAIMA/BBB_ES/BBB_ES.csv")
anyDuplicated(base_BBB_ES)
anyNA(base_BBB_ES)
str(base_BBB_ES)
dim(base_BBB_ES)#580 202
summary(base_BBB_ES$name)#eliminar
table(base_BBB_ES$class) #Positive= 304 Negative= 276 IR=1.1  304/276

require(tidyverse)
prop.table(table(base_BBB_ES$class)) %>% round(digits = 2)# negative=0.48    positive= 0.52

require(caret)
nearZeroVar(base_BBB_ES)
base_BBB_ES<-base_BBB_ES[,-1]


#SE DEBE PARTICIONAR LA BASE EN UN CONJUNTO DE PRUEBA Y ENTRENAMIENTO.
set.seed(987654)
entrenamientoBBB_ES<- createDataPartition(y = base_BBB_ES$class, p = 0.8, list = FALSE, times = 1)
base_entrenamientoBBB_ES <- base_BBB_ES[entrenamientoBBB_ES, ]
base_pruebaBBB_ES  <- base_BBB_ES[-entrenamientoBBB_ES, ]

nearZeroVar(base_entrenamientoBBB_ES)
nearZeroVar(base_pruebaBBB_ES)#

table(base_entrenamientoBBB_ES$class) #positive= 244 negative=221 IR=1.1
table(base_pruebaBBB_ES$class) #positive=60  negative=55 IR=1.1
prop.table(table(base_entrenamientoBBB_ES$class))#negative=0.4752688  positive=0.5247312  nivel basal
prop.table(table(base_pruebaBBB_ES$class))#negative=0.4782609 positive=0.5217391   nivel basal

3setwd("/media/ubuntu/YANAIMA/BBB_ES")
write.csv(base_entrenamientoBBB_ES,"base_entrenamientoBBB_ES.csv")
write.csv(base_pruebaBBB_ES,"base_pruebaBBB_ES.csv")

#A partir de aquí minar

library(doParallel)
library(kernlab)
library(caret)

cl <- makeCluster(4)
registerDoParallel(cl, cores=4)
getDoParWorkers()

seeds <- vector(mode = "list", length = 51)
for(i in 1:50) seeds[[i]] <- seeds[[i]] <- seq(100, 999, by = 92) 
## For the last model:
seeds[[51]] <- 514

ctrl <- trainControl(method="repeatedcv", number=5, repeats=10, seeds = seeds,  
                     allowParallel = TRUE, savePredictions="final", returnResamp = "final",
                     classProbs=TRUE) #returnResamp = "final"

set.seed(123)
earthFit.BBB_ES <- train(class ~ . , 
                         data = base_entrenamientoBBB_ES,
                         method = "earth", 
                         preProcess=c("center","scale"),
                         tuneGrid = data.frame(degree = 1, nprune = (1:10) * 3),
                         trControl =ctrl)
earthFit.BBB_ES$bestTune
#setwd("/media/ubuntu/YANAIMA/BBB_ES/")
confusionMatrix.train(earthFit.BBB_ES)#0.7211
# write.csv(earthFit.BBB_ES$results, "earthFit.BBB_ES_results.csv")
# write.csv(rbind(confusionMatrix.train(earthFit.BBB_ES)), "CMTrainingearthFit.BBB_ES.csv")
saveRDS(earthFit.BBB_ES,"earthFit.BBB_ES.rds")
varImp(earthFit.BBB_ES)
test_earthFit.BBB_ES<-predict(earthFit.BBB_ES, newdata=base_pruebaBBB_ES)
cm_test_earthFit.BBB_ES<- confusionMatrix(test_earthFit.BBB_ES ,base_pruebaBBB_ES$class)#Accuracy :  0.7391304 
# Error de test
error_test_earth_BBB_ES <- mean(test_earthFit.BBB_ES != base_pruebaBBB_ES$class)
paste("El error de test del modelo:", round(error_test_earth_BBB_ES*100, 2), "%")#1-Accuracy
cm_test_earthFit.BBB_ES$overall[1]#0.7391304 

predicciones_BBB_ES <- extractPrediction(
  models = list(earth = earthFit.BBB_ES,rf=rfFit_BBB_ES),
  testX = base_pruebaBBB_ES %>% select(-class),
  testY = base_pruebaBBB_ES$class
)
predicciones_BBB_ES %>% head()
###########
set.seed(123)
rfFit_BBB_ES<- train(class ~ . , data = base_entrenamientoBBB_ES, method = "rf", ntree = 300, 
                     tuneGrid = expand.grid(mtry = c(round(round(sqrt(ncol(base_entrenamientoBBB_ES) - 1)) / 2), round(sqrt(ncol(base_entrenamientoBBB_ES) - 1)), 2 * round(sqrt(ncol(base_entrenamientoBBB_ES) - 1)))), #2,4,8
                     preProcess = c("center", "scale"),
                     trControl = ctrl, importance = TRUE,proximity = TRUE)#do.trace=TRUE,
summary(rfFit_BBB_ES$result) #0.78
rfFit_BBB_ES$finalModel
plot(rfFit_BBB_ES)
varImpPlot(rfFit_BBB_ES$finalModel)
round(max(rfFit_BBB_ES$results$Accuracy),2) #0.78
confusionMatrix.train(rfFit_BBB_ES)#0.7813
saveRDS(rfFit_BBB_ES,"rfFit.BBB_ES.rds")

test_rfFit_BBB_ES<-predict(rfFit_BBB_ES, newdata=base_pruebaBBB_ES)
cm_rfFit_BBB_ES<- confusionMatrix(data=test_rfFit_BBB_ES ,base_pruebaBBB_ES$class)
cm_rfFit_BBB_ES$overall[1]#Acc   0.7826087 

#
library(randomForest)
library(ranger)
modelLookup("ranger")
#mtry: número predictores seleccionados aleatoriamente en cada árbol.
#min.node.size: tamaño mínimo que tiene que tener un nodo para poder ser dividido.
#splitrule: criterio de división.
set.seed(123)
modelo_rangerFit.BBB_ES<-train(class ~ ., data = base_entrenamientoBBB_ES, #selectionFunction="best", 
                               method = "ranger",
                               preProcess=c("center","scale"),
                               tuneGrid = expand.grid(mtry =  c(2,4,8),
                                                      min.node.size = 2,splitrule = c("hellinger","gini","extratrees")),#gini para clasif  #min.node.size = c(2,5,7)
                               trControl = ctrl, num.trees = 60, importance='impurity')#importance='permutation' importance='impurity' Gini #0.7766374 ACC
summary(modelo_rangerFit.BBB_ES$results) #0.7856   ACC
modelo_rangerFit.BBB_ES$finalModel
confusionMatrix(modelo_rangerFit.BBB_ES)#0.7856
plot(modelo_rangerFit.BBB_ES)
varImp(modelo_rangerFit.BBB_ES)
saveRDS(modelo_rangerFit.BBB_ES,"rangerFit.BBB_ES.rds")

test_modelo_ranger.BBB_ES<-predict(modelo_rangerFit.BBB_ES, newdata=base_pruebaBBB_ES)
cm_modelo_ranger.BBB_ES<- confusionMatrix(data=test_modelo_ranger.BBB_ES ,base_pruebaBBB_ES$class)
cm_modelo_ranger.BBB_ES$overall[1]#Acc 0.7652174  
#

set.seed(123)
knnFit_BBB_ES <- train(class ~ . , data = base_entrenamientoBBB_ES,
                       method = "knn", 
                       preProcess=c("center","scale"),
                       tuneGrid=expand.grid(k=1:10),
                       trControl = ctrl)
summary(knnFit_BBB_ES$results) #
summary(mean(knnFit_BBB_ES$resample$Accuracy))#promedio 
round(max(knnFit_BBB_ES$results$Accuracy),2) #0.71
confusionMatrix.train(knnFit_BBB_ES)#0.7135
knnFit_BBB_ES$finalModel

#setwd("/media/ubuntu/YANAIMA/BBB_ES/")
# write.csv(knnFit_BBB_ES$results, "knnFit_BBB_ES_results.csv")
# write.csv(knnFit_BBB_ES$bestTune, "knnFit_BBB_ESbestTune.csv")
# write.csv(rbind(confusionMatrix.train(knnFit_BBB_ES)), "CMTrainingknnFit_BBB_ES.csv")
saveRDS(knnFit_BBB_ES,"knnFit_BBB_ES.rds")

test_knnFit_BBB_ES<-predict(knnFit_BBB_ES, newdata=base_pruebaBBB_ES)
cm_knnFit_BBB_ES<- confusionMatrix(data=test_knnFit_BBB_ES ,base_pruebaBBB_ES$class)#0.6869565  
confusionMatrix(test_knnFit_BBB_ES, base_pruebaBBB_ES$class)$table
cm_knnFit_BBB_ES$overall[1]#  0.6869565   
##
set.seed(123) 
kknnFit.BBB_ES <- train(class ~ .,
                        method = "kknn", 
                        data = base_entrenamientoBBB_ES, #preProcess=c("center","scale"),
                        tuneGrid=expand.grid(kmax = 10, 
                                             distance = (1:4), 
                                             kernel = "optimal"),
                        trControl =ctrl)# 
confusionMatrix.train(kknnFit.BBB_ES)#0.7204
saveRDS(kknnFit.BBB_ES,"kknnFit_BBB_ES.rds")

test_kknnFit.BBB_ES<-predict(kknnFit.BBB_ES, newdata=base_pruebaBBB_ES)
cm_test_kknnFit.BBB_ES<- confusionMatrix(test_kknnFit.BBB_ES ,base_pruebaBBB_ES$class)#Accuracy 
cm_test_kknnFit.BBB_ES$overall[1]#0.7391304
#
modelLookup("avNNet")
set.seed(123)
avnnetFit.BBB_ES <- train(class ~., data=base_entrenamientoBBB_ES, method="avNNet", trace=F,maxit = 150,
                          tuneGrid=expand.grid(decay=c(0.01,0.1,1), size=c(1,4,5), bag=FALSE),
                          preProcess=c("center","scale"),
                          trControl = ctrl)#
confusionMatrix.train(avnnetFit.BBB_ES)#0.754
saveRDS(avnnetFit.BBB_ES,"avnnetFit.BBB_ES.rds")

test_avnnetFit.BBB_ES<-predict(avnnetFit.BBB_ES, newdata=base_pruebaBBB_ES)
cm_test_avnnetFit.BBB_ES<- confusionMatrix(test_avnnetFit.BBB_ES ,base_pruebaBBB_ES$class)#Accuracy 0.7652174 
cm_test_avnnetFit.BBB_ES$overall[1]#0.6869565 
#
set.seed(123)
treeFit_BBB_ES<- train(class ~ ., data = base_entrenamientoBBB_ES,
                       method = "treebag",
                       nbagg = 50,preProcess=c("center","scale"),
                       trControl = ctrl)
confusionMatrix(treeFit_BBB_ES) #0.7662 
treeFit_BBB_ES$finalModel
saveRDS(treeFit_BBB_ES,"treeFit_BBB_ES.rds")

test_treeFit_BBB_ES<-predict(treeFit_BBB_ES, newdata=base_pruebaBBB_ES)
cm_treeFit_BBB_ES<- confusionMatrix(data=test_treeFit_BBB_ES ,base_pruebaBBB_ES$class)
cm_treeFit_BBB_ES$overall[1]# 0.7217391  

#
set.seed(123)
nbFit_BBB_ES <- train(class ~ . , data = base_entrenamientoBBB_ES,
                      method = "nb", preProcess=c("center","scale"),
                      tuneGrid = expand.grid(fL=c(0:3),usekernel=c(FALSE,TRUE), adjust=1),
                      trControl = ctrl)
summary(mean(nbFit_BBB_ES$resample$Accuracy))#promedio  
round(max(nbFit_BBB_ES$results$Accuracy),2) #0.7166

saveRDS(nbFit_BBB_ES,"nbFit_BBB_ES.rds")

test_nb_BBB_ES<-predict(nbFit_BBB_ES,base_pruebaBBB_ES)
cm_nb_BBB_ES<- confusionMatrix(data=test_nb_BBB_ES ,base_pruebaBBB_ES$class)
cm_nb_BBB_ES$overall[1]#0.6869565
#

require(caret)
library(nnet)#Method Value:nnetfrom packagennetwith tuning parameters:size,decay
nnetGrid_BBB_ES=expand.grid(decay=c(0.01,0.1,1), size=c(1,3,5)) 
set.seed(123)
nnetFit_BBB_ES <- train(class ~ . , data = base_entrenamientoBBB_ES, method="nnet", trace=F,maxit = 200,
                        tuneGrid=nnetGrid_BBB_ES,preProcess=c("center","scale"),#rang = c(-0.8, 0.8),
                        trControl = ctrl)
nnetFit_BBB_ES$finalModel
varImp(nnetFit_BBB_ES)
confusionMatrix.train(nnetFit_BBB_ES)#0.7533
saveRDS(nnetFit_BBB_ES,"nnetFit_BBB_ES.rds")

test_nnetFit_BBB_ES<-predict(nnetFit_BBB_ES, newdata=base_pruebaBBB_ES)
cm_nnetFit_BBB_ES<- confusionMatrix(data=test_nnetFit_BBB_ES ,base_pruebaBBB_ES$class)
cm_nnetFit_BBB_ES$overall#Acc 0.678260869

#OJO gbm SE USA PARA REGRESION Y PARA CLASIFICACION Method Value:gbmfrom packagegbmwith tuning parameters:interaction.depth,n.trees,shrinkage
set.seed(123)
system.time(gbmFit_BBB_ES <- train(class ~ ., data = base_entrenamientoBBB_ES, 
                                   method = "gbm", 
                                   trControl = ctrl,
                                   tuneGrid=expand.grid(n.trees = (1:30)*50, interaction.depth = c(1,5, 9), shrinkage = c(0.002,0.02) ,n.minobsinnode = 7),
                                   verbose = FALSE)
)

trellis.par.set(caretTheme())
plot(gbmFit_BBB_ES)  
summary(gbmFit_BBB_ES$results)
gbmFit_BBB_ES$finalModel$interaction.depth

confusionMatrix.train(gbmFit_BBB_ES)#0.7899
saveRDS(gbmFit_BBB_ES,"gbmFit_BBB_ES.rds")

test_gbmFit_BBB_ES<-predict(gbmFit_BBB_ES, newdata=base_pruebaBBB_ES)
cm_gbmFit_BBB_ES<- confusionMatrix(data=test_gbmFit_BBB_ES ,base_pruebaBBB_ES$class)
cm_gbmFit_BBB_ES$overall#Acc 7.565217 


#########################
# set.seed(123)#no se pudo, SI CON BBB_AC
# ldaFit_BBB_ES <- train(class ~., data = base_entrenamientoBBB_ES, method = "lda",
#                                  trControl = ctrl)
# test_ldaFit_BBB_ES<-predict(ldaFit_BBB_ES, newdata=base_pruebaBBBCN)
# cm_ldaFit_BBB_ES<- confusionMatrix(data=test_ldaFit_BBB_ES ,base_pruebaBBBCN$class)
# #
set.seed(123)
modelo_C50.BBB_ES <- train(class ~ ., data = base_entrenamientoBBB_ES,
                           method = "C5.0", 
                           trControl = ctrl,tuneGrid = expand.grid(model = c("tree","rules"),
                                                                   trials = c(10,12,14),
                                                                   winnow = c(FALSE, TRUE))
)  
round(max(modelo_C50.BBB_ES$results$Accuracy),2)#0.75
confusionMatrix(modelo_C50.BBB_ES)#0.7542
plot(modelo_C50.BBB_ES)
saveRDS(modelo_C50.BBB_ES,"modelo_C50.BBB_ES.rds")

test_c50Fit_BBB_ES<-predict(modelo_C50.BBB_ES, newdata=base_pruebaBBB_ES)
cm_c50Fit_BBB_ES<- confusionMatrix(data=test_c50Fit_BBB_ES ,base_pruebaBBB_ES$class)#0.7565217  
cm_c50Fit_BBB_ES$overall[1]#0.7217391 
#############
set.seed(123)
system.time(svmLinearFit_BBB_ES <- train(class ~ . , data = base_entrenamientoBBB_ES,
                                         method = "svmLinear", 
                                         tuneGrid=expand.grid(C=c(0.001,0.01,0.1,0.02,0.03,0.04,0.05,0.06)),
                                         trControl = ctrl))

round(max(svmLinearFit_BBB_ES$results$Accuracy),2) #0.76
confusionMatrix.train(svmLinearFit_BBB_ES)#0.7572
svmLinearFit_BBB_ES$finalModel
saveRDS(svmLinearFit_BBB_ES,"svmLinearFit_BBB_ES.rds")

test_svmLinearFit_BBB_ES<-predict(svmLinearFit_BBB_ES, newdata=base_pruebaBBB_ES)
cm_svmLinearFit_BBB_ES<- confusionMatrix(data=test_svmLinearFit_BBB_ES ,base_pruebaBBB_ES$class)
cm_svmLinearFit_BBB_ES$overall[1]#0.6695652 

confusionMatrix(test_svmLinearFit_BBB_ES, base_pruebaBBB_ES$class)$table
confusionMatrix(test_svmLinearFit_BBB_ES, base_pruebaBBB_ES$class)$overall[1]#0.6695652
###########
#
set.seed(123) 
svmPolyFit_BBB_ES <- train(class ~ .,
                           method = "svmPoly", 
                           data = base_entrenamientoBBB_ES, 
                           #tuneLength=3,
                           tuneGrid=expand.grid(scale=2, degree=c(1,2), C=c(0.001,0.01,0.1,1)),
                           trControl = ctrl)
round(max(svmPolyFit_BBB_ES$results$Accuracy),2) #0.74
confusionMatrix(svmPolyFit_BBB_ES)
svmPolyFit_BBB_ES$finalModel
saveRDS(svmPolyFit_BBB_ES,"svmPolyFit_BBB_ES.rds")

test_svmPoly_BBB_ES<-predict(svmPolyFit_BBB_ES, newdata=base_pruebaBBB_ES)
cm_svmPoly_BBB_ES<- confusionMatrix(data=test_svmPoly_BBB_ES ,base_pruebaBBB_ES$class)
cm_svmPoly_BBB_ES$overall[1]#Acc 0.7043478 
#########
#sigma: coeficiente del kernel radial.  C: penalización por violaciones del margen del hiperplano.
set.seed(123)
svmRadialFit_BBB_ES <- train(class ~ ., data = base_entrenamientoBBB_ES,
                             method = "svmRadial", preProcess = c("center", "scale"),
                             trControl = ctrl,
                             tuneGrid=expand.grid(sigma = c(.01, .1, .2, 1),C=c(0.1,1)),
                             prob.model = TRUE)
svmRadialFit_BBB_ES$finalModel
confusionMatrix.train(svmRadialFit_BBB_ES)#0.7353
saveRDS(svmRadialFit_BBB_ES,"svmRadialFit_BBB_ES.rds")

test_svmRadialFit_BBB_ES<-predict(svmRadialFit_BBB_ES, newdata=base_pruebaBBB_ES)
cm_svmRadial_BBB_ES<- confusionMatrix(data=test_svmRadialFit_BBB_ES ,base_pruebaBBB_ES$class)
cm_svmRadial_BBB_ES$overall[1]#Acc 0.6956522  
#########

# #modelo_C50Tree no tiene parametros
set.seed(123)
modelo_C50Tree.BBB_ES <- train(class ~ ., data = base_entrenamientoBBB_ES,
                               method = "C5.0Tree", #no tiene parametro
                               trControl = ctrl)
summary(modelo_C50Tree.BBB_ES$finalModel)
summary(modelo_C50Tree.BBB_ES$results)#0.6776
confusionMatrix(modelo_C50Tree.BBB_ES)
saveRDS(modelo_C50Tree.BBB_ES,"modelo_C50Tree.BBB_ES.rds")

test_c50TreeFit_BBB_ES<-predict(modelo_C50Tree.BBB_ES,base_pruebaBBB_ES)
cm_c50TreeFit_BBB_ES<- confusionMatrix(data=test_c50TreeFit_BBB_ES ,base_pruebaBBB_ES$class)
paste("El error de test del modelo:",round(mean(test_c50TreeFit_BBB_ES!=base_pruebaBBB_ES$class)*100,2),"%")#37.39%
cm_c50TreeFit_BBB_ES$overall[1]#0.626087
################
#cp complexity parameter paramtero de complejidad, param positivo, q a menor valor hace q sea mas profundo el arbol y mas complejo y puede provocar sobreajuste
set.seed(123)
rpartFit_BBB_ES <- train(class ~ . , data = base_entrenamientoBBB_ES, method = "rpart",
                         trControl = ctrl,tuneGrid =expand.grid(cp = seq(0,0.5,0.06) ))
#tuneLength = 10)
summary(rpartFit_BBB_ES$results)
rpartFit_BBB_ES$finalModel$cptable
confusionMatrix.train(rpartFit_BBB_ES)#0.6869
summary(mean(rpartFit_BBB_ES$resample$Accuracy))#promedio 0.6905
round(max(rpartFit_BBB_ES$results$Accuracy),2) #0.69

plot(rpartFit_BBB_ES$finalModel)
rpart.plot::rpart.plot(rpartFit_BBB_ES$finalModel, main="Àrbol RPART en BBB_AC con RFE")
ggplot(rpartFit_BBB_ES) 
rpartFit_BBB_ES$bestTune$cp
rpart.plot::rpart.plot(rpartFit_BBB_ES$finalModel, main="Modelo con el algoritmo RPART en la base bd_entrenamiento BBB_AC con RFE")
plot(rpartFit_BBB_ES$finalModel,margin=0.1)
text(rpartFit_BBB_ES$finalModel,cex = 0.75)
library(rattle)
fancyRpartPlot(rpartFit_BBB_ES$finalModel)

saveRDS(rpartFit_BBB_ES,"rpartFit_BBB_ES.rds")

cm_rpartFit.BBB_ES<- confusionMatrix(data=predict(rpartFit_BBB_ES,base_pruebaBBB_ES) ,base_pruebaBBB_ES$class)# 
print(confusionMatrix(data=predict(rpartFit_BBB_ES,base_pruebaBBB_ES) ,base_pruebaBBB_ES$class)$overall["Accuracy"])# Accuracy 0.6521739 

