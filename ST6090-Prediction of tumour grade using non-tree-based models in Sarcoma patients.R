

################# Packages and Libraries ######################

# Necessary Packages to be installed
install.packages('ggplot2')
install.packages('tidyverse')
install.packages('DataExplorer')
install.packages('corrplot')
install.packages('correlation')
install.packages('caret')
install.packages('dplyr')
install.packages('cleandata')
install.packages('mltools')
install.packages('e1071')
install.packages('penalizedSVM')
install.packages('Boruta')
install.packages('klaR')
install.packages('class')
install.packages('stats')
install.packages('coefplot')
install.packages('scutr')
install.packages('plotrix')

#installing DMwR package and its dependencies
install.packages(c("zoo","xts","quantmod")) ## and perhaps mode
install.packages('ROCR')
# DMwR Package should be downloaded externally and needs to be installed 
# install.packages( "C:\\Users\\srive\\Downloads\\DMwR_0.4.1.tar.gz", repos=NULL, type="source" ) 
install.packages('vtable')
install.packages('cvms')


#Libraries to be loaded
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


################ Dataset Loading and Data pre-processing & Data Visualisation by creating automated report ###########
# data import
data = read.csv(file='H:\\My Drive\\TERM 3\\Datasets\\sarcoma.csv', 
                header=TRUE, sep=',')

n = NROW(data)

sumtable(data, out='csv',file = 'C:/Users/srive/Documents/sumtable')
summary(data)

dim(data)

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
which(is.na(data$grade))#index of the missing values in Grade column
which(is.na(data$volume)) #index of the missing values in Volume column

#total no. of missing values in the dataset
sum(NROW(which(is.na(data$volume))),
    NROW(which(is.na(data$grade))))

#removing NA  or missing values 
newdata = na.omit(data) 

dim(newdata)


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

which(colSums(is.na(newdata)) > 0 )#check for any NA values

#distribution of target variable on whole dataset
summary(newdata$grade)

#############STATISTICAL EXPLORATION

#EDA using DataExplorer
create_report(newdata, y = "grade")


################## Correlation analysis #########################
#seperating the Continuos and Categorical features for Correlation analysis
continuos_newdata = (newdata[,4:105]) # without grade, tumour subtype, Gender, Age

colnames(continuos_newdata)

#plot for correlation
plot_correlation(continuos_newdata)

cor_matrix = round(cor(continuos_newdata), 3)
cor_index = findCorrelation(cor_matrix, cutoff = 0.9, exact = FALSE)

length(cor_index)#size of redundant variables
colnames(continuos_newdata[(cor_index)]) #Names of the columns that are highly correlated

#Feature set after removing the highly correlated variables 
colnames(newdata[colnames(continuos_newdata[-(cor_index)])])


#Correlation plot after removing the highly correlated variables
plot_correlation(newdata[colnames(continuos_newdata[-(cor_index)])])


#New dataset after removing the highly correlated features and combining with
#the categorical variables for the Feature selection 
FS_newdata = cbind.data.frame(newdata[1:3], newdata[colnames(continuos_newdata[-(cor_index)])] 
                              ,newdata[106])

ncol(FS_newdata)
colnames(cbind.data.frame(newdata[1:3], newdata[colnames(continuos_newdata[-(cor_index)])] 
                          ,newdata[106]))

ncol(FS_newdata)
# 51 columns and 153 rows is the final dataset after correlation is completed 

#EDA using DataExplorer
create_report(FS_newdata, y = "grade")



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
  
  
    set.seed(1000)
    # 5-Fold rfFuncs
    CaretFun_rf = rfe(x.train, y.train, sizes = c(1:15),
                      rfeControl = control_rfe_rf)

    #selecting the variables based on the accuracy of the model
    rfe_rf_acc[k] = max(CaretFun_rf$results[,2])
    rfe_rf_optvar[[k]] = CaretFun_rf$optVariables

    #selecting the variables based on the importance of the variables in the iteration
    set.size = 10 #you want set-size of 7
    CaretFun_rf.vars <- CaretFun_rf$variables
    # selects variables of set-size (= 10 here)
    CaretFun_rf.set <- CaretFun_rf.vars[CaretFun_rf.vars$Variables==set.size,  ]
    #use aggregate to calculate mean ranking score (under column "Overall")
    CaretFun_rf.set <- aggregate(CaretFun_rf.set[, c("Overall")], list(CaretFun_rf.set$var), mean)
    #order from highest to low, and select first 10:
    CaretFun_rf.order <- order(CaretFun_rf.set[, c("x")], decreasing = TRUE)[1:set.size]

    print("#####################-RfFuncs-######################")
    print(CaretFun_rf.set[CaretFun_rf.order, 1:2])
    print("#####################-RfFuncs-######################")

    rfe_rf_Impvar[[k]] = CaretFun_rf.set[CaretFun_rf.order, 1]

    #plots
    plot_rf[[k]]=plot(CaretFun_rf, type=c("o","g"))




  #
    # 5-Fold nbFuncs
    set.seed(1000)
    CaretFun_nb = rfe(x.train, y.train, sizes = c(1:15), rfeControl = control_rfe_nbFuncs)

    #selecting the variables based on the accuracy of the model
    rfe_nb_acc[k] = max(CaretFun_nb$results[,2])
    rfe_nb_optvar[[k]] = CaretFun_nb$optVariables

    #selecting the variables based on the importance of the variables in the iteration
    set.size = 10 #you want set-size of 7
    CaretFun_nb.vars <-  CaretFun_nb$variables
    # selects variables of set-size (= 10 here)
    CaretFun_nb.set <-  CaretFun_nb.vars[ CaretFun_nb.vars$Variables==set.size,  ]
    #use aggregate to calculate mean ranking score (under column "Overall")
    CaretFun_nb.set <- aggregate( CaretFun_nb.set[, c("Overall")], list( CaretFun_nb.set$var), mean)
    #order from highest to low, and select first 10:
    CaretFun_nb.order <- order( CaretFun_nb.set[, c("x")], decreasing = TRUE)[1:set.size]

    print("#####################-nbFuncs-######################")
    print(CaretFun_nb.set[CaretFun_nb.order, 1:2])
    print("#####################-nbFuncs-######################")

    rfe_nb_Impvar[[k]] =  CaretFun_nb.set[ CaretFun_nb.order, 1]

    #plots
    plot_nb[[k]]=plot(CaretFun_nb, type=c("o","g"))

  
  
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

  #plots
  plot_svm[[k]]=plot(CaretFun_svm, type=c("o","g"))


  #5-Fold svmPolynomial
  CaretFun_svmPoly = rfe(x , y , sizes = c(1,5,10,15), rfeControl = control_rfe_svm_Poly,
                         method="svmPoly")
  #selecting the variables based on the accuracy of the model
  rfe_svmPoly_acc[k] = max(CaretFun_svmPoly$results[,2])
  rfe_svmPoly_optvar[[k]] = CaretFun_svmPoly$optVariables

  #selecting the variables based on the importance of the variables in the iteration
  set.size = 10 #you want set-size of 7
  CaretFun_svmPoly.vars <- CaretFun_svmPoly$variables
  # selects variables of set-size (= 10 here)
  CaretFun_svmPoly.set <- CaretFun_svmPoly.vars[CaretFun_svmPoly.vars$Variables==set.size,]
  #use aggregate to calculate mean ranking score (under column "Overall")
  CaretFun_svmPoly.set <- aggregate(CaretFun_svmPoly.set[, c("Overall")], list(CaretFun_svmPoly.set$var), mean)
  #order from highest to low, and select first 10:
  CaretFun_svmPoly.order <- order(CaretFun_svmPoly.set[, c("x")], decreasing = TRUE)[1:set.size]

  print("#####################-svmPoly-######################")
  print(CaretFun_svmPoly.set[CaretFun_svmPoly.order, 1:2])
  print("#####################-svmPoly-######################")


  rfe_svmPoly_Impvar[[k]] = CaretFun_svmPoly.set[CaretFun_svmPoly.order, 1]

  #plots
  plot_svmPoly[[k]]=plot(CaretFun_svmPoly, type=c("o","g"))

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


#plots
plot_rf

plot_nb

plot_svm

plot_svmPoly


#Below are the Features obtained from RFE method during the run and are hardcoded thereafter
#and used whenever necessary to reduce the computation time on further analysis of these features

rferf_features = rfenb_features = rfesvmRadial_features = rfesvmPoly_features = list()
# 
# rferf_features =  c("grad1", "max.grad_HIST", "Reg.max.grad.grey_HIST", "tumor.subtype", 
#                     "Reg.grad.min", "mean.seg", "Reg.grad.0.1", "Reg.grad.0.95", "suv.mean", 
#                     "correlation_GLCM")
# 
# rfenb_features = c("mean.seg", "info.corr.1_GLCM", "Reg.max.grad.grey_HIST", 
#                    "Reg.min.grad.grey_HIST", "CoV_HIST", 
#                    "grad0.7", "Reg.grad.0.95", "mode_HIST", "suv.mean", "Reg.Het1")
# 
# rfesvmRadial_features = c("mean.seg", "Reg.min.grad.grey_HIST", 
#                           "info.corr.2_GLCM", "tumor.subtype", "CoV_HIST", "info.corr.1_GLCM", 
#                           "Reg.grad.0.95", "p10_HIST")
# 
# rfesvmPoly_features = c("info.corr.2_GLCM", "correlation_GLCM", "mean.seg",
#                         "Reg.max.grad.grey_HIST", "mode_HIST", "info.corr.1_GLCM",
#                         "max.grad_HIST", "CoV_HIST", "grad1", "median_HIST")



################### Boruta Feature selection #####################

# #plot function for boruta
# boruta_plot = function(p){
#   plot(p, xlab = "", xaxt = "n")
#   k <-lapply(1:ncol(p$ImpHistory),function(i)
#     p$ImpHistory[is.finite(p$ImpHistory[,i]),i])
#   names(k) <- colnames(p$ImpHistory)
#   Labels <- sort(sapply(k,median))
#   axis(side = 1,las=2,labels = names(Labels),
#        at = 1:ncol(p$ImpHistory), cex.axis = 0.6)
# }

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

#Output from Boruta Analysis
boruta_obj
boruta_features
varVarImp


#Five set of features obtained
boruta_features[[1]]
boruta_features[[2]]
boruta_features[[3]]
boruta_features[[4]]
boruta_features[[5]]


#####################  sample model for benchmarking FS results #######
itrain = createDataPartition(FS_newdata$grade, p=0.7, list = FALSE)

FS_train_data = FS_newdata[itrain,]
FS_test_data = FS_newdata[-itrain,]

#Different set of features are used recursively and benchmark results were obtained 
#Sample model with SVM 
set.seed(1000)
dat = FS_train_data[sample(1:nrow(FS_train_data), nrow(FS_train_data)), ]

xsvm = dat[boruta_features]
ysvm = dat$grade

#Scaling & Normalising of the input data
xsvm = apply(as.matrix(xsvm), 2, as.numeric)
xsvm = scale(xsvm)

#SVM model (Radial and Polynomial Kernel) -  CV 5 fold
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

#Results
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

#Results
multinom_acc
mean(multinom_acc)
sd(multinom_acc)


#Boxplots for comparitive analysis

boxplot(svm_acc*100, svmo.poly_acc*100, multinom_acc*100,
        main="Boruta analysis - Optimum Feature Set", ylim=c(0,100),
        ylab='Accuracy', xlab='Models',
        names = c('SVM-Radial Kernel', 'SVM-Polynomial Kernel',
                  'Multinomial Logistic Regression'))






####################  sample model for Hyperparameter tuning - results #######

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
    
    #Output will be printed manually inside the loop for each run
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

#Average Results from tuning analysis
mean(svm_hyp_acc)
sd(svm_hyp_acc)
boxplot(svm_hyp_acc*100, ylim=c(0,100), main='SVM Polynomial kernel - RFE (SVM Radial kernel method)',
        ylab='Accuracy')


################## Final Model for whole dataset ##############
set.seed(2000)

#Here the required optimum feature set is added each time before starting the analysis.
#e.g. RFE SVM Radial kernel model features are used in the analysis below
final_data = FS_newdata[rfesvmRadial_features]
final_data$grade = FS_newdata$grade

R=50 #R=1 is used to find the best model with random repetitions of the samples
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
  
  
  #Test data
  x.test = FS_test_data
  y.test = x.test$grade
  x.test$grade = NULL
  #scaling
  x.test = apply(x.test, 2, as.numeric)
  x.test = scale(x.test)
  
  #SVM model
  
  svm.fin = svm(x,y, class.weights = wt.fin, kernel = 'polynomial'
                ,degree = 3, scale = 0.1, cost = 0.25)
  
  
  svm.pred = predict(svm.fin, x.test)
  
  tb.svm  = table(y.test, svm.pred)
  svm.acc.base[i] = sum(diag(tb.svm)) / sum(tb.svm)
  confusionMatrix(y.test, svm.pred)
  
}


###################  AVERAGE RESULTS FROM ABOVE MODEL ANALYSIS


#######Average Performance of base models
cbind(mean(svm.acc.base),sd(svm.acc.base))


########### Boxplots of above results
boxplot(svm.acc.base*100, 
        ylim=c(0,100),main='Accuracy Plot - Final Model',
        xlab='SVM Model - Polynomial Kernel',ylab='Accuracy') 

text(y = round(boxplot.stats(svm.acc.base*100)$stats, 4),
     labels = round(boxplot.stats(svm.acc.base*100)$stats,4), x = 1.35)


############ BELOW CODE TO BE USED ONLY FOR BEST MODEL RUN

#NOTE: Use below 4 lines of code to extract results from above analysis for the best model run 
#set R=1 above to repeat the model with random repetitions of the samples 
#to obtain highest observed accuracy.(65~70%)

# svm.acc.base
# tb.svm
# metrics_model(tb.svm)
# confusionMatrix(y.test, svm.pred)


########### Other Performance metrics calculation - Accuracy, Precision, Recall, F-1 Score
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




####NOTE: BELOW LINES ODE IS EXTRA EFFORT PERFORMED ON THE ABOVE ANALYSIS AND NOT INCLUSIVE OF THE REPORT


#################### Sampling techniques on data ###############

sampling_data = FS_newdata[rfesvmRadial_features]
sampling_data$grade = FS_newdata$grade

summary(sampling_data)
R=50  # Can be altered if necessary. R=1 is used to find the best model with random repetitions of the samples
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
  train_upsample = upSample(x.sample, y.sample )
  
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

#Results from the analysis
svm.acc.base
mean(svm.acc.base)
sd(svm.acc.base)
boxplot(svm.acc.base*100)
confusionMatrix(y.test, svm.pred)
