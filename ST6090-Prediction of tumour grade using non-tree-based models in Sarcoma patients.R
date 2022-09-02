
par(mfrow=c(1,1))
version

################# Intro and Libraries ######################

# Packages
# install.packages('ggplot2')
# install.packages('tidyverse')
# install.packages('DataExplorer')
# install.packages('corrplot')
# install.packages('correlation')
# install.packages('caret')
# install.packages('dplyr')
# install.packages('cleandata')
# install.packages('mltools')
# install.packages('e1071')
# install.packages('penalizedSVM')
# install.packages('Boruta')
# install.packages('klaR')
# install.packages('class')
# install.packages('stats')
# install.packages('coefplot')
# install.packages("BBmisc") # for normalisation of the features
# install.packages('scutr')
# install.packages('plotrix')

#installing DMwR package and its dependencies
# install.packages(c("zoo","xts","quantmod")) ## and perhaps mode
# install.packages('ROCR')
# install.packages( "C:\\Users\\srive\\Downloads\\DMwR_0.4.1.tar.gz", repos=NULL, type="source" )
install.packages('vtable')
install.packages('cvms')


#libraries
library(e1071)
library(caret)
library(mltools)
library(data.table)
library(corrplot)
library(DataExplorer)
library(Boruta)
library(klaR)
library(class)
library(base)
library(glmnet)
library(pROC)
library(LiblineaR)
library(nnet)
library(stats)
library(BBmisc)
library(scutr)
library(DMwR)
library(plotrix)
library(vtable)
library(cvms)


dim(data)
# data import
data = read.csv(file='H:\\My Drive\\TERM 3\\Datasets\\sarcoma.csv', 
                header=TRUE, sep=',')

n = NROW(data)
nrow(data)

sumtable(data, out='csv',file = 'C:/Users/srive/Documents/sumtable')
summary(data)


ncol(data)
# Data exploration
# converting character to factor datatype for categorical variables
data$grade = as.factor(data$grade)
data$sex = as.factor(data$sex)
data$tumor.subtype = as.factor(data$tumor.subtype)
data[,1] = NULL # removing the first column

# removing surtim,surind,dfstim,dfsind features. since these are not required in 
# detecting the tumour grade as the main aim does not deals with the prognosis data
data[,107:110] = NULL 


#missing data
pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(data,2,pMiss)# percentage of missing data across columns


which(colSums(is.na(data)) > 0 ) # columns with more than one NA values
which(is.na(data$grade))
which(is.na(data$volume))

#total no. of missing values in the dataset
sum(NROW(which(is.na(data$volume))),
    NROW(which(is.na(data$grade)))
)

NROW(data)

# removing NA values 
newdata = na.omit(data) 
NROW(newdata)

NCOL(newdata)


####Recoding the categorical variables
#ordinal variable encodings

#variable - sex
newdata$sex = factor(newdata$sex, levels = c('Male', 'Female'), 
                     labels = c(1,0))

#variable - tumor type
newdata$tumor.subtype = factor(newdata$tumor.subtype, levels = c('Bone', 'STS', 'Cartilage'), 
                               labels = c(1,2,3))

#ordinal variable encoding
newdata$grade = factor(newdata$grade,levels = c('high', 'int', 'low'),
                       labels = c(3,2,1))

which(colSums(is.na(newdata)) > 0 )#no NA values

#distribution of target variable on whole dataset
summary(newdata$grade)

dim(newdata)
#############STATISTICAL EXPLORATION

#EDA using DataExplorer
# create_report(newdata, y = "grade")

############################## PCA Anlaysis - START ##############################
##PCA ANALYSIS
###########################  TRAIN TEST SPLIT  
itrain = sample(1:n, size = round(0.7*NROW(newdata)), replace=FALSE)

train_data = newdata[itrain,]
test_data = newdata[-itrain,]

x.train = train_data[c(2:106)]
x.test = test_data[c(2:106)]

y.train = train_data[1]
y.test = test_data[1]

colnames(x.train)

#training &test data for PCA analysis
pca.x.train = x.train
pca.x.test = x.test

#conversion of factor to numeric for PCA 
pca.x.train$sex = as.numeric(x.train$sex)
pca.x.train$tumor.subtype = as.numeric(x.train$tumor.subtype)

pca.x.test$sex = as.numeric(x.test$sex)
pca.x.test$tumor.subtype = as.numeric(x.test$tumor.subtype)

#applying PCA
prcomp_PCA = prcomp(pca.x.train, center = TRUE, scale = TRUE)

names(prcomp_PCA)

summary(prcomp_PCA)

head(prcomp_PCA$x[,1:12])


#screeplot
screeplot(prcomp_PCA, type = "l", npcs = 20, main = "Screeplot of the PCs")
abline(h = 1, col="red", lty=5)
legend("topright", legend=c("Eigenvalue = 12"),
       col=c("red"), lty=5, cex=0.6)

#cumulative variance plot
cumpro <- cumsum(prcomp_PCA$sdev^2 / sum(prcomp_PCA$sdev^2))
plot(cumpro[0:20], xlab = "PC #", ylab = "Amount of explained variance", 
     main = "Cumulative variance plot")
abline(v = 12, col="blue", lty=5) # choose eigen value for 90% variablity
abline(h = 0.90534, col="blue", lty=5)
legend("topleft", legend=c("Cut-off @ PC12"),
       col=c("blue"), lty=5, cex=0.6)


#PCA data preparation for Classification algorithms
pca.train_data = predict(prcomp_PCA, newdata = pca.x.train)
pca.test_data = predict(prcomp_PCA, newdata = pca.x.test)

# PCA data for models: 
# Train - pca.train_data
# Test - pca.test_data


#build models using the PCA vs default dataset and compare the performances
#discuss about the other feature selection methods (filters, wrappers, embedded
#methods)

pca.train_data = as.data.frame(pca.train_data)
pca.test_data = as.data.frame(pca.test_data)

pca.train_data = pca.train_data[,1:12]
pca.test_data = pca.test_data[,1:12]

#PCA components data frame size differs - Should check on that

#NEED TO LOOK ON THIS SVM ISSUE

#SVM using PCA train & test data
svm.pca = svm( as.factor(y.train) ~., pca.train_data )
# svmo.pred = predict(svm.pca, newdata = pca.test_data)



############################## PCA Anlaysis - END 


################## Correlation analysis & RFE  - START #########################


#seperating the Radiomic and Clinical features for Correlation analysis
continuos_newdata = (newdata[,4:105]) # without grade, tumour subtype, Gender, Age

colnames(continuos_newdata)

#plot for correlation
plot_correlation(continuos_newdata)

cor_matrix = round(cor(continuos_newdata), 3)
cor_index = findCorrelation(cor_matrix, cutoff = 0.9, exact = FALSE)

#Names of the columns that are highly correlated
length(cor_index)
colnames(continuos_newdata[(cor_index)])

#Feature set after removing the highly correlated variables 
colnames(newdata[colnames(continuos_newdata[-(cor_index)])])


#Correlation plot after removing the highly correlated variables
plot_correlation(newdata[colnames(continuos_newdata[-(cor_index)])])


#New dataset after removing the highly correlated features and 
#the categorical variables for the RFE 
FS_newdata = cbind.data.frame(newdata[1:3], newdata[colnames(continuos_newdata[-(cor_index)])] 
                              ,newdata[106])

ncol(FS_newdata)
colnames(cbind.data.frame(newdata[1:3], newdata[colnames(continuos_newdata[-(cor_index)])] 
                          ,newdata[106]))

ncol(FS_newdata)
# 51 columns and 153 rows is the final dataset after correlation is completed 

#EDA using DataExplorer
# create_report(FS_newdata, y = "grade")

#################Inner resampling scheme - 5 fold on WHOLE DATASET for correlated features ######## 

#
set.seed(1000)

itrain = createDataPartition(FS_newdata$grade, p=0.7, list = FALSE)

FS_train_data = FS_newdata[itrain,]
FS_test_data = FS_newdata[-itrain,]

#Weights
wts = 10 / table(FS_train_data$grade)
model_weights <- ifelse(FS_train_data$grade == 3, wts[1],
                        (ifelse(FS_train_data$grade == 2, wts[2], wts[3])))

multi.acc = numeric()
#SVM Model on Correlation removed features
trc_5 = trainControl(method="repeatedcv", number = 5, repeats = 5 )

x=as.data.frame(FS_train_data[,-1])
y=(FS_train_data[,1])

#5 fold- multi
Multilog.fit = train(x,y, method = "multinom",
                     trControl = trc_5, weights = model_weights)

Multilog.pred = predict(Multilog.fit, newdata = FS_test_data)
tb = table(FS_test_data$grade, Multilog.pred)
multi.acc = sum(diag(tb)) / sum(tb)


confusionMatrix(FS_test_data$grade, Multilog.pred)
tb
multi.acc

summary(FS_train_data$grade)
summary(FS_test_data$grade)
############ RFE using RandomForest & Naive Bayes & SVM using Radial and Polynomial Kernel#################
set.seed(1000)
itrain = createDataPartition(FS_newdata$grade, p=0.7, list = FALSE)

FS_train_data = FS_newdata[itrain,]
FS_test_data = FS_newdata[-itrain,]

#repeated 5-fold CV
control_rfe_rf = rfeControl(functions = rfFuncs, 
                            method = "repeatedcv", 
                            number = 5,
                            repeats = 5)

control_rfe_nbFuncs = rfeControl(functions = nbFuncs, 
                                 method = "repeatedcv", 
                                 number = 5,
                                 repeats = 5)

control_rfe_svm = rfeControl(functions = caretFuncs,
                             method = "repeatedcv", 
                             number = 5,
                             repeats = 5)

control_rfe_svm_Poly = rfeControl(functions = caretFuncs,
                                  method = "repeatedcv", 
                                  number = 5,
                                  repeats = 5)


K = 5
n = nrow(FS_train_data)
folds = cut(1:n, K, labels=FALSE)

rfe_rf_acc = rfe_nb_acc = rfe_svm_acc = numeric(K)
rfe_rf_optvar = rfe_nb_optvar = rfe_svm_optvar = list()
rfe_rf_Impvar = rfe_nb_Impvar = rfe_svm_Impvar = list()
plot_rf = plot_nb = plot_svm = list()

#svmPoly
rfe_svmPoly_acc = numeric(K)
rfe_svmPoly_optvar = rfe_svmPoly_Impvar = plot_svmPoly = list()

set.seed(1000)
dat_rfe = FS_train_data[sample(1:nrow(FS_train_data), nrow(FS_train_data)), ]

cl <- makeCluster(4)    # 4 core-machine
registerDoParallel(cl)

for(k in 1:K){
  
  i.train = which(folds!=k)
  
  x.train = dat_rfe[i.train,]
  y.train = x.train$grade
  x.train$grade = NULL
  
  df.rfe = cbind.data.frame(y.train, x.train)
  i.test = which(folds==k)
  
  
  #   set.seed(1000)
  #   # 5-Fold rfFuncs
  #   CaretFun_rf = rfe(x.train, y.train, sizes = c(1:15),
  #                     rfeControl = control_rfe_rf)
  # 
  #   #selecting the variables based on the accuracy of the model
  #   rfe_rf_acc[k] = max(CaretFun_rf$results[,2])
  #   rfe_rf_optvar[[k]] = CaretFun_rf$optVariables
  # 
  #   #selecting the variables based on the importance of the variables in the iteration
  #   set.size = 10 #you want set-size of 7
  #   CaretFun_rf.vars <- CaretFun_rf$variables
  #   # selects variables of set-size (= 10 here)
  #   CaretFun_rf.set <- CaretFun_rf.vars[CaretFun_rf.vars$Variables==set.size,  ]
  #   #use aggregate to calculate mean ranking score (under column "Overall")
  #   CaretFun_rf.set <- aggregate(CaretFun_rf.set[, c("Overall")], list(CaretFun_rf.set$var), mean)
  #   #order from highest to low, and select first 10:
  #   CaretFun_rf.order <- order(CaretFun_rf.set[, c("x")], decreasing = TRUE)[1:set.size]
  # 
  #   print("#####################-RfFuncs-######################")
  #   print(CaretFun_rf.set[CaretFun_rf.order, 1:2])
  #   print("#####################-RfFuncs-######################")
  # 
  #   rfe_rf_Impvar[[k]] = CaretFun_rf.set[CaretFun_rf.order, 1]
  # 
  #   #plots
  #   plot_rf[[k]]=plot(CaretFun_rf, type=c("o","g"))
  # 
  # 
  # 
  # 
  # #
  #   # 5-Fold nbFuncs
  #   set.seed(1000)
  #   CaretFun_nb = rfe(x.train, y.train, sizes = c(1:15), rfeControl = control_rfe_nbFuncs)
  # 
  #   #selecting the variables based on the accuracy of the model
  #   rfe_nb_acc[k] = max(CaretFun_nb$results[,2])
  #   rfe_nb_optvar[[k]] = CaretFun_nb$optVariables
  # 
  #   #selecting the variables based on the importance of the variables in the iteration
  #   set.size = 10 #you want set-size of 7
  #   CaretFun_nb.vars <-  CaretFun_nb$variables
  #   # selects variables of set-size (= 10 here)
  #   CaretFun_nb.set <-  CaretFun_nb.vars[ CaretFun_nb.vars$Variables==set.size,  ]
  #   #use aggregate to calculate mean ranking score (under column "Overall")
  #   CaretFun_nb.set <- aggregate( CaretFun_nb.set[, c("Overall")], list( CaretFun_nb.set$var), mean)
  #   #order from highest to low, and select first 10:
  #   CaretFun_nb.order <- order( CaretFun_nb.set[, c("x")], decreasing = TRUE)[1:set.size]
  # 
  #   print("#####################-nbFuncs-######################")
  #   print(CaretFun_nb.set[CaretFun_nb.order, 1:2])
  #   print("#####################-nbFuncs-######################")
  # 
  #   rfe_nb_Impvar[[k]] =  CaretFun_nb.set[ CaretFun_nb.order, 1]
  # 
  #   #plots
  #   plot_nb[[k]]=plot(CaretFun_nb, type=c("o","g"))
  
  
  
  #5-Fold svmRadial
  x.train = dat_rfe[i.train,]
  xm = apply(as.matrix(x.train), 2, as.numeric)
  
  #scaling
  y = xm[,1]
  x = scale(xm[,2:ncol(xm)])
  
  y = factor(y,levels = c(3,2,1), labels = c('high', 'int', 'low'))
  
  CaretFun_svm = rfe(x , y , sizes = c(1:15),
                     rfeControl = control_rfe_svm, method="svmRadial", metric ="Accuracy")
  #selecting the variables based on the accuracy of the model
  rfe_svm_acc[k] = max(CaretFun_svm$results[,2])
  rfe_svm_optvar[[k]] = CaretFun_svm$optVariables
  
  #selecting the variables based on the importance of the variables in the iteration
  set.size = 10 #you want set-size of 7
  CaretFun_svm.vars <- CaretFun_svm$variables
  # selects variables of set-size (= 10 here)
  CaretFun_svm.set <- CaretFun_svm.vars[CaretFun_svm.vars$Variables==set.size,]
  #use aggregate to calculate mean ranking score (under column "Overall")
  CaretFun_svm.set <- aggregate(CaretFun_svm.set[, c("Overall")], list(CaretFun_svm.set$var), mean)
  #order from highest to low, and select first 10:
  CaretFun_svm.order <- order(CaretFun_svm.set[, c("x")], decreasing = TRUE)[1:set.size]
  
  print("#####################-svmFuncs-######################")
  print(CaretFun_svm.set[CaretFun_svm.order, 1:2])
  print("#####################-svmFuncs-######################")
  
  
  rfe_svm_Impvar[[k]] = CaretFun_svm.set[CaretFun_svm.order, 1]
  # 
  # #plots
  # plot_svm[[k]]=plot(CaretFun_svm, type=c("o","g"))
  
  
  #5-Fold svmPolynomial
  # CaretFun_svmPoly = rfe(x , y , sizes = c(1,5,10,15), rfeControl = control_rfe_svm_Poly,
  #                        method="svmPoly")
  # #selecting the variables based on the accuracy of the model
  # rfe_svmPoly_acc[k] = max(CaretFun_svmPoly$results[,2])
  # rfe_svmPoly_optvar[[k]] = CaretFun_svmPoly$optVariables
  # 
  # #selecting the variables based on the importance of the variables in the iteration
  # set.size = 10 #you want set-size of 7
  # CaretFun_svmPoly.vars <- CaretFun_svmPoly$variables
  # # selects variables of set-size (= 10 here)
  # CaretFun_svmPoly.set <- CaretFun_svmPoly.vars[CaretFun_svmPoly.vars$Variables==set.size,]
  # #use aggregate to calculate mean ranking score (under column "Overall")
  # CaretFun_svmPoly.set <- aggregate(CaretFun_svmPoly.set[, c("Overall")], list(CaretFun_svmPoly.set$var), mean)
  # #order from highest to low, and select first 10:
  # CaretFun_svmPoly.order <- order(CaretFun_svmPoly.set[, c("x")], decreasing = TRUE)[1:set.size]
  # 
  # print("#####################-svmPoly-######################")
  # print(CaretFun_svmPoly.set[CaretFun_svmPoly.order, 1:2])
  # print("#####################-svmPoly-######################")
  # 
  # 
  # rfe_svmPoly_Impvar[[k]] = CaretFun_svmPoly.set[CaretFun_svmPoly.order, 1]
  
  #plots
  # plot_svmPoly[[k]]=plot(CaretFun_svmPoly, type=c("o","g"))
  
}

#Best results and Features from RFE-rfFuncs
rfe_rf_acc
rfe_rf_optvar
rfe_rf_Impvar

which.max(rfe_rf_acc)
rfe_rf_acc[which.max(rfe_rf_acc)]
rfe_rf_optvar[[which.max(rfe_rf_acc)]][1:10]
rfe_rf_Impvar[which.max(rfe_rf_acc)]

#Best results and Features from RFE-nbFuncs
rfe_nb_acc
rfe_nb_optvar
rfe_nb_Impvar

which.max(rfe_nb_acc)
rfe_nb_acc[which.max(rfe_nb_acc)]
rfe_nb_optvar[[which.max(rfe_nb_acc)]][1:10]
rfe_nb_Impvar[which.max(rfe_nb_acc)]


#Best results and Features from RFE-svmRadial
rfe_svm_acc
rfe_svm_optvar
rfe_svm_Impvar

which.max(rfe_svm_acc)
rfe_svm_acc[which.max(rfe_svm_acc)]
rfe_svm_optvar[[which.max(rfe_svm_acc)]][1:10]
rfe_svm_Impvar[which.max(rfe_svm_acc)]

#Best results and Features from RFE-svmPoly
rfe_svmPoly_acc
rfe_svmPoly_optvar
rfe_svmPoly_Impvar

which.max(rfe_svmPoly_acc)
rfe_svmPoly_acc[which.max(rfe_svmPoly_acc)]
rfe_svmPoly_optvar[[which.max(rfe_svmPoly_acc)]][1:10]
rfe_svmPoly_Impvar[which.max(rfe_svmPoly_acc)]

?rfe

#plots
plot_rf

plot_nb

plot_svm

plot_svmPoly


#Below are the Features obtained from RFE method
rferf_features = rfenb_features = rfesvmRadial_features = rfesvmPoly_features = list()

rferf_features =  c("grad1", "max.grad_HIST", "Reg.max.grad.grey_HIST", "tumor.subtype", 
                    "Reg.grad.min", "mean.seg", "Reg.grad.0.1", "Reg.grad.0.95", "suv.mean", 
                    "correlation_GLCM")

rfenb_features = c("mean.seg", "info.corr.1_GLCM", "Reg.max.grad.grey_HIST", 
                   "Reg.min.grad.grey_HIST", "CoV_HIST", 
                   "grad0.7", "Reg.grad.0.95", "mode_HIST", "suv.mean", "Reg.Het1")

rfesvmRadial_features = c("mean.seg", "Reg.min.grad.grey_HIST", 
                          "info.corr.2_GLCM", "tumor.subtype", "CoV_HIST", "info.corr.1_GLCM", 
                          "Reg.grad.0.95", "p10_HIST")

rfesvmPoly_features = c("info.corr.2_GLCM", "correlation_GLCM", "mean.seg",
                        "Reg.max.grad.grey_HIST", "mode_HIST", "info.corr.1_GLCM",
                        "max.grad_HIST", "CoV_HIST", "grad1", "median_HIST")


################### Boruta Feature selection #####################
#Boruta analysis

#plot function for boruta
boruta_plot = function(p){
  plot(p, xlab = "", xaxt = "n")
  k <-lapply(1:ncol(p$ImpHistory),function(i)
    p$ImpHistory[is.finite(p$ImpHistory[,i]),i])
  names(k) <- colnames(p$ImpHistory)
  Labels <- sort(sapply(k,median))
  axis(side = 1,las=2,labels = names(Labels),
       at = 1:ncol(p$ImpHistory), cex.axis = 0.6)
}

#
set.seed(1000)
itrain = createDataPartition(FS_newdata$grade, p=0.7, list = FALSE)

FS_train_data = FS_newdata[itrain,]
FS_test_data = FS_newdata[-itrain,]

#5-fold CV
set.seed(1000)
K = 5
n = nrow(FS_train_data)
folds = cut(1:n, K, labels=FALSE)
boruta_obj = boruta_features = bor.log.acc = varVarImp = list()

for(k in 1:K){
  set.seed(1000)
  #X, Y
  i.train = which(folds!=k)
  x = FS_train_data[i.train,]
  y = x$grade
  x$grade = NULL
  df = cbind.data.frame(y,x)
  
  i.test = which(folds==k)
  x.test = FS_train_data[i.test,]
  y.test = x.test$grade
  x.test$grade = NULL
  
  boruta_fs = Boruta(y ~ ., data = df, doTrace = 2, maxRuns = 200 )
  tentative.boruta = TentativeRoughFix(boruta_fs)
  
  boruta_obj[[k]] = tentative.boruta
  boruta_features[[k]] = getSelectedAttributes(tentative.boruta, withTentative = F)
  
  #plots for each run
  # boruta_plot(tentative.boruta)
  
  #print variableImp score 
  i = which(attStats(tentative.boruta)[6]=='Confirmed')
  varimpscore = attStats(tentative.boruta)[i,c('maxImp')]
  attributes = getSelectedAttributes(tentative.boruta)
  
  varVarImp[[k]] = (cbind.data.frame(attributes, varimpscore))
  
}

boruta_obj
boruta_features
varVarImp

boruta_features[[1]]
boruta_features[[2]]
boruta_features[[3]]
boruta_features[[4]]
boruta_features[[5]]

######### Feature selection using the LASSO regression method ###############################

set.seed(1000)
itrain = createDataPartition(FS_newdata$grade, p=0.7, list = FALSE)

Lasso_train_data = FS_newdata[itrain,]
Lasso_test_data = FS_newdata[-itrain,]

#5-fold CV for fitting the glm model and extracting the coef
set.seed(1000)
K = 5
n = nrow(Lasso_train_data)
folds = cut(1:n, K, labels=FALSE)
coef_lasso = list()
opt_lambda = list()


for(k in 1:K){
  i.train = which(folds!=k)
  x=Lasso_train_data[i.train,]
  y=x$grade
  x$grade=NULL
  
  i.test = which(folds==k)
  
  xm = apply(as.matrix(x), 2, as.numeric)
  
  CV = cv.glmnet(xm, y, family='multinomial', alpha=1,  nfolds = 5)
  
  opt_lambda[[k]] = CV$lambda.1se
  
  fit = glmnet(xm, y, family='multinomial', alpha=1, lambda=CV$lambda.1se)
  
  coef_lasso[[k]] = fit$beta
  
}

opt_lambda
coef_lasso

## should deal with the class imbalance data and modelling of them



#####################  sample model for benchmarking FS results #######
itrain = createDataPartition(FS_newdata$grade, p=0.7, list = FALSE)

FS_train_data = FS_newdata[itrain,]
FS_test_data = FS_newdata[-itrain,]

# boruta_features = c("tumor.subtype", "grad1", "suv.mean", 
#                       "mean.seg", "max.grad_HIST", "Reg.max.grad.grey_HIST", "Reg.grad.0.95")

#Sample model with SVM 
set.seed(1000)
dat = FS_train_data[sample(1:nrow(FS_train_data), nrow(FS_train_data)), ]

xsvm = dat[boruta_features]
ysvm = dat$grade

#Scaling & Normalising of the input data
xsvm = apply(as.matrix(xsvm), 2, as.numeric)
xsvm = scale(xsvm)
# xsvm = normalize(xsvm, method = "range", range = c(0, 1))


#SVM model -  CV 5 fold
K = 5
n = nrow(xsvm)
folds = cut(1:n, K, labels=FALSE)
svm_acc = svmo.poly_acc = numeric()

for (k in 1:K){
  
  i.train = which(folds!=k)
  i.test = which(folds==k)
  
  #train
  x.train = xsvm[i.train,]
  y.train = ysvm[i.train]
  #test
  x.test = xsvm[i.test,]
  y.test = ysvm[i.test]
  
  #Weights
  wts = 100 / table(y.train)
  
  #model- svm Radial
  svmo = svm(x.train, y.train, kernel = "radial", class.weights = wts )
  svmo.pred = predict(svmo, x.test)
  tb=table(y.test,svmo.pred)
  svm_acc[k] = sum(diag(tb)) / sum(tb)
  
  #model- svmPolynomial
  svmo.poly = svm(x.train, y.train, kernel = "polynomial", class.weights = wts )
  svmo.poly.pred = predict(svmo.poly, x.test)
  tb.poly=table(y.test,svmo.poly.pred)
  svmo.poly_acc[k] = sum(diag(tb.poly)) / sum(tb.poly)
  
}

mean(svm_acc)
sd(svm_acc)


mean(svmo.poly_acc)
sd(svmo.poly_acc)

################################Multinomial logistic regression 
itrain = createDataPartition(FS_newdata$grade, p=0.7, list = FALSE)

FS_train_data = FS_newdata[itrain,]
FS_test_data = FS_newdata[-itrain,]
dat = FS_train_data[sample(1:nrow(FS_train_data), nrow(FS_train_data)), ]

xmul = dat[boruta_features]
ymul = dat$grade

#Scaling & Normalising of the input data
xmul = apply((xmul), 2, as.numeric)
xmul = scale(xmul)

#model building
#Multinom model -  CV 5 fold
K = 5
n = nrow(xmul)
folds = cut(1:n, K, labels=FALSE)
multinom_acc = numeric()

for (k in 1:K){
  i.train = which(folds!=k)
  i.test = which(folds==k)
  
  #train
  x.train = xmul[i.train,]
  y.train = ymul[i.train]
  #test
  x.test = xmul[i.test,]
  y.test = ymul[i.test]
  
  dat.train = cbind.data.frame(y.train,x.train)
  
  #Weights
  wts = 100 / table(y.train)
  model_weights <- ifelse(y.train == 3, wts[1],
                          (ifelse(y.train == 2, wts[2], wts[3])))
  
  #model
  mul.o = multinom(y.train~., data=dat.train, weights = model_weights)
  mulo.pred = predict(mul.o, x.test)
  
  tb=table(y.test,mulo.pred)
  multinom_acc[k] = sum(diag(tb)) / sum(tb)
  
}


multinom_acc
mean(multinom_acc)
sd(multinom_acc)



# 
# boxplot(svm_acc*100, svmo.poly_acc*100, multinom_acc*100, 
#         main="Boruta analysis - Optimum Feature Set", ylim=c(0,100), 
#         ylab='Accuracy', xlab='Models', 
#         names = c('SVM-Radial Kernel', 'SVM-Polynomial Kernel', 
#                   'Multinomial Logistic Regression'))
# 
# #Avg acccuracies
mean(svm_acc)
mean(svmo.poly_acc)
mean(multinom_acc)



#S.E
sd(svmo.poly_acc)
sd(svm_acc)
# sd(multinom_acc)


#####################  sample model for Hyperparameter tuning - results #######

set.seed(1000)
Hyp_data = FS_newdata[rferf_features]
Hyp_data$grade = FS_newdata$grade

itrain = createDataPartition(Hyp_data$grade, p=0.7, list = FALSE)

Hyp_train_data = Hyp_data[itrain,]
Hyp_test_data = Hyp_data[-itrain,]


#Add repeat here for introducing random splits in a repeat fashion
trc_tune = trainControl(method = 'repeatedcv', number=5, repeats = 5) 


R=5
K=5
n = nrow(Hyp_train_data)
folds = cut(1:n, K, labels=FALSE)
svm_hyp_acc = numeric(K*R)
# Hyp_parameters_cost = Hyp_parameters_gamma = Hyp_parameters_error = numeric(K)


for (r in 1:R){
  set.seed(1000)
  Hyp_train_data = Hyp_train_data[sample(1:nrow(Hyp_train_data), nrow(Hyp_train_data), replace = FALSE), ]
  xsvm = Hyp_train_data
  ysvm = Hyp_train_data$grade
  xsvm$grade = NULL
  
  #Scaling & Normalising of the input data
  xsvm = apply(as.matrix(xsvm), 2, as.numeric)
  xsvm = scale(xsvm)
  
  for (k in 1:K){
    
    i.train = which(folds!=k)
    i.test = which(folds==k)
    
    #train
    x.train = xsvm[i.train,]
    y.train = ysvm[i.train]
    train.df = cbind.data.frame(x.train,y.train)
    #test
    x.test = xsvm[i.test,]
    y.test = ysvm[i.test]
    
    tune_svm = train(y.train~., data = train.df, method="svmPoly",
                     trControl=trc_tune, preProcess = c("center","scale"),
                     tuneLength = 5)
    
    print('##############################################################################')
    print(r)
    print(k)
    print(cbind.data.frame(k,tune_svm$bestTune$degree,tune_svm$bestTune$scale,tune_svm$bestTune$C))
    print('##############################################################################')
    
    #svm-model
    wts = 100 / table(y.train)
    
    svm_hyp = svm(x.train, y.train, kernel = "polynomial", class.weights = wts,
                  degree = tune_svm$bestTune$degree, scale = tune_svm$bestTune$scale,
                  cost = tune_svm$bestTune$C)
    
    svm_hyp.pred = predict(svm_hyp, newdata=x.test)
    
    tb=table(y.test,svm_hyp.pred)
    svm_hyp_acc[k+(r-1)*K] = sum(diag(tb)) / sum(tb)
    
  }
  
}

length(svm_hyp_acc)
mean(svm_hyp_acc)
sd(svm_hyp_acc)
boxplot(svm_hyp_acc*100, ylim=c(0,100), main='SVM Polynomial kernel - RFE (SVM Radial kernel method)',
        ylab='Accuracy')
# tune_parameters_best = cbind.data.frame(Hyp_parameters_cost, Hyp_parameters_gamma, Hyp_parameters_error)
# tune_parameters_best = as.matrix.data.frame(tune_parameters_best)
# 
# cat( paste(tune_parameters_best$Hyp_parameters_cost , collapse='\n'))
# cat( paste(tune_parameters_best$Hyp_parameters_gamma , collapse='\n' )) 
# cat( paste( round(tune_parameters_best$Hyp_parameters_error,4) , collapse='\n' ) )


################## Final Model for whole dataset ##############
set.seed(2000)


final_data = FS_newdata[rfesvmRadial_features]
final_data$grade = FS_newdata$grade


R=1
svm.acc.base = numeric(R)

for (i in 1:R){
  #Random sampling the data
  final_data = final_data[sample(1:nrow(final_data), nrow(final_data)),]
  itrain = createDataPartition(final_data$grade, p=0.7, list = FALSE)
  
  FS_train_data = final_data[itrain,]
  FS_test_data = final_data[-itrain,]
  
  #Train data
  x = FS_train_data
  y = x$grade
  x$grade = NULL
  #scaling
  x = apply(x, 2, as.numeric)
  x = scale(x)
  
  df=cbind.data.frame(x,y)
  #weights
  wt.fin = 100 / table(y)
  
  svm.weights = ifelse(y == 3, wt.fin[1], (ifelse(y == 2, wt.fin[2], wt.fin[3])))
  
  
  #tuning
  # C <- c(0.25,0.5,1,4)
  # degree <- c(1,2,3)
  # scale <- c(0.1,0.01,1)
  # gr.poly <- expand.grid(C=C,degree=degree,scale=scale)
  # 
  #Test data
  x.test = FS_test_data
  y.test = x.test$grade
  x.test$grade = NULL
  #scaling
  x.test = apply(x.test, 2, as.numeric)
  x.test = scale(x.test)
  
  #SVM model
  
  
  # svm.fin = train(y~., data = df, method="svmPoly", weights = svm.weights,
  #                 tuneGrid=gr.poly)
  
  svm.fin = svm(x,y, class.weights = wt.fin, kernel = 'polynomial'
                ,degree = 3, scale = 0.1, cost = 0.25)
  
  
  svm.pred = predict(svm.fin, x.test)
  
  tb.svm  = table(y.test, svm.pred)
  svm.acc.base[i] = sum(diag(tb.svm)) / sum(tb.svm)
  confusionMatrix(y.test, svm.pred)
  
}


svm.acc.base
tb.svm
metrics_model(tb.svm)
# confusionMatrix(y.test, svm.pred)


########## Performance of base models
cbind(mean(svm.acc.base),sd(svm.acc.base))

########### Boxplots of above results
boxplot(svm.acc.base*100, 
        ylim=c(0,100),main='Accuracy Plot - Final Model',
        xlab='SVM Model - Polynomial Kernel',ylab='Accuracy') 

text(y = round(boxplot.stats(svm.acc.base*100)$stats, 4),
     labels = round(boxplot.stats(svm.acc.base*100)$stats,4), x = 1.35)



########### Other metrics calculation
metrics_model = function(tb){
  n = sum(tb) # number of instances
  nc = nrow(tb) # number of classes
  diag = diag(tb) # number of correctly classified instances per class 
  rowsums = apply(tb, 1, sum) # number of instances per class
  colsums = apply(tb, 2, sum) # number of predictions per class
  p = rowsums / n # distribution of instances over the actual classes
  q = colsums / n # distribution of instances over the predicted
  
  #Accuracy
  accuracy = sum(diag) / n
  print(accuracy)
  
  #Per-class Precision, Recall, and F-1
  precision = diag / colsums 
  recall = diag / rowsums 
  f1 = 2 * precision * recall / (precision + recall) 
  print(data.frame(precision, recall, f1))
  
  #Macro-averaged Metrics
  macroPrecision = mean(precision)
  macroRecall = mean(recall)
  macroF1 = mean(f1)
  print(data.frame(macroPrecision, macroRecall, macroF1))
}


#################### Sampling techniques on data ###############

# rferf_features = rfenb_features = rfesvmRadial_features = rfesvmPoly_features = list()

sampling_data = FS_newdata[rfesvmRadial_features]
sampling_data$grade = FS_newdata$grade

summary(sampling_data)

R=1
svm.acc.base = numeric(R)

for (r in 1:R){
  set.seed(1000)
  sampling_data = sampling_data[sample(1:nrow(sampling_data), nrow(sampling_data)),]
  itrain = createDataPartition(sampling_data$grade, p=0.7, list = FALSE)
  
  sampling_data_train = sampling_data[itrain,]
  sampling_data_test = sampling_data[-itrain,]
  
  #sampling x and y
  x.sample = sampling_data_train
  y.sample = x.sample$grade
  x.sample$grade = NULL
  
  #random oversampling technique
  train_upsample = downSample(x.sample, y.sample )
  
  #model
  x = train_upsample
  y = x$Class
  x$Class = NULL
  #scaling
  x = apply(x, 2, as.numeric)
  x = scale(x)
  df=cbind.data.frame(x,y)
  
  #Test data
  x.test = sampling_data_test
  y.test = x.test$grade
  x.test$grade = NULL
  #scaling
  x.test = apply(x.test, 2, as.numeric)
  x.test = scale(x.test)
  
  #SVM model
  # trC_sampling = trainControl(method = 'repeatedcv', number=5, repeats = 5) 
  
  svm.fin = train(y~., data = df, method="svmPoly",
                  preProcess = c("center","scale"), tuneLength = 3)
  
  # svm.fin = svm(x,y, kernel = 'polynomial')
  
  svm.pred = predict(svm.fin, x.test)
  
  tb.svm = table(y.test, svm.pred)
  
  svm.acc.base[r]= sum(diag(tb.svm)) / sum(tb.svm)
  
}

svm.acc.base
mean(svm.acc.base)
sd(svm.acc.base)
boxplot(svm.acc.base*100)
confusionMatrix(y.test, svm.pred)$byClass


cor(FS_newdata[rfesvmRadial_features])



newdata[105]
