#----------------------------------------------------------------------------------------------#
#                            Universidad del Valle - Escuela de Estadística                    #
#                                         Programa Académico de:                               #
#                                        Pregrado en Estadística                               #
#                Asignatura : Técnicas de Minería de Datos y Aprendizaje Automático            #           
#                                  Profesor - Jaime Mosquera Restrepo                          #
#----------------------------------------------------------------------------------------------#
# Estudiantes: Yeimy Tatiana Marín código:1524344-3752 -                                       #
#              Miguel Enriquez     código:2023796-334                                          #
#----------------------------------------------------------------------------------------------#
#                         0. Configuración inicial-Librerías requeridas                     ####
#----------------------------------------------------------------------------------------------#
wd="C:\\Users\\yeimy\\Downloads\\lab4"       # Ruta al Directorio de trabajo
setwd(wd)                                    # Establecer el directorio de trabajo 

#install.packages("easypackages")        # Libreria especial para hacer carga automática de librerias
library("easypackages")

# Listado de librerias requeridas por el script
lib_req<-c("MASS","car","epiR","visdat","corrplot","FactoMineR","factoextra",
           "caret","pls","rpart","rpart.plot","e1071", "randomForest","doBy")
# Verificación, instalación y carga de librerias.
easypackages::packages(lib_req)         

#----------------------------------------------------------------------------------------------#
#              Aprendizaje Supervisado - Regresión                                          ####
#              Caso - Diabetes()                                                               #
#----------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------#
## 1. Lectura de Datos, visualización y transformación                                      ####
#----------------------------------------------------------------------------------------------#

# Lectura de Datos en R
diabetes=data.frame(read.table("diabetes.txt", header = TRUE))

# Visualización de los datos
View(diabetes); str(diabetes)

# Transformación de las variables categóricas a factor
diabetes=transform(diabetes,
                   SEX=factor(SEX,levels=1:2,labels=c("1","2"))
                   )

# Visualización de datos faltantes
windows(height=10,width=15)
visdat::vis_miss(diabetes)   

nobs.comp =sum(complete.cases(diabetes))      # Cuenta los registros completos
nobs.miss = sum(!complete.cases(diabetes))    # Cuenta los registros con datos faltantes.

#----------------------------------------------------------------------------------------------#
## 2. Análisis descriptivo de las variables                                                 ####
#----------------------------------------------------------------------------------------------#

# Descripción de las variables
summary(diabetes)

# Visualización general de la Variable Respuesta Y.
windows(height=10,width=15)
M=matrix(c(1,1,1,1,1,1,2,2,2),byrow=T,ncol=3)
layout(M)
with(diabetes,{
  hist(Y, col="Blue",freq=F,breaks=25,xlim=range(Y))  
  lines(density(Y))
  boxplot(Y,horizontal=T, col="Blue",ylim=range(Y))
})

# Boxplots individuales de todas las variables, excepto la variable Y. 
windows(height=10,width=15)
par(mfrow=c(2,5)) 
tabla<-prop.table(table(diabetes$SEX))
coord<-barplot(tabla, col=c("gray","Blue"),ylim=c(0,1.1), main="SEX")
text(coord,tabla,labels=round(tabla,2), pos=3)
lapply(names(diabetes[,-c(2,11)]),function(y){
  (boxplot(diabetes[,y],ylab= y,boxwex = 0.5,col="Blue"))
})

# Relación bivariada con la variable respuesta Y. 
color=c("Gray", "Blue")    # Define los tipos de sexo (1 y 2) por colores para diferenciarlos.

attach(diabetes)
windows(height=10,width=15)
par(mfrow=c(2,5)) 
boxplot(Y~SEX,col=color)
lapply(colnames(diabetes[,-c(2,11)]),function(y){
  plot(diabetes[SEX=="1",y],diabetes[SEX=="1","Y"],col=color[1],pch=20,ylab="Y",xlab=y)
  points(diabetes[SEX=="2",y],diabetes[SEX=="2","Y"],col=color[2],pch=20)
  lines(smooth.spline(diabetes[,y],diabetes[,"Y"],df=3),col="Black")
})   
detach(diabetes)

# Matriz de correlación para todas las variables cuantitativas
AQ.cor = cor(diabetes[,-2],method="pearson")
print(AQ.cor)
windows(height=10,width=15)
corrplot::corrplot(AQ.cor, method = "ellipse",addCoef.col = "black",type="upper")

# Visualización Multivariada apoyada en Componentes principales.
Modelo_PCA <- FactoMineR::PCA(diabetes,ncp=3,quali.sup=2)
# Valores propios y varianza acumulada
Modelo_PCA$eig
VP=Modelo_PCA$eig[,1]; Var= Modelo_PCA$eig[,2]; Var_acum=Modelo_PCA$eig[,3]  

windows(height=10,width=15)
par(mfrow=c(1,2))
coord=barplot(VP, xlab="Componente",ylab="Valor Propio", ylim=c(0,max(VP)+1))
lines(coord,VP,col="blue",lwd=2)
text(coord,VP,paste(round(Var,2),"%"), pos=3,cex=0.6)
abline(h=1,col="red", lty=2)
coord=barplot(Var_acum, xlab="Componente",ylab="Varianza Acumulada")
lines(coord,Var_acum,col="blue",lwd=2)
text(coord,Var_acum,round(Var_acum,2), pos=3,cex=0.6)

# Visualización de la Correlación entre variables
windows(height=10,width=15)
factoextra::fviz_pca_ind(Modelo_PCA ,habillage=diabetes$SEX,col.ind =color,addEllipses = T, ellipse.level = 0.95) 
windows(height=10,width=15)
factoextra::fviz_pca_biplot(Modelo_PCA ,axes = c(1, 2),habillage=diabetes$SEX,col.ind =color) 
windows(height=10,width=15)
factoextra::fviz_pca_biplot(Modelo_PCA ,axes = c(2, 3),habillage=diabetes$SEX,col.ind =color) 
windows(height=10,width=15)
factoextra::fviz_pca_biplot(Modelo_PCA ,axes = c(1, 3),habillage=diabetes$SEX,col.ind =color) 

#----------------------------------------------------------------------------------------------#
## 3. Entrenamiento y validación de  modelos de regresión                                   ####
#----------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------#
# 3.0. División de datos entrenamiento/test                                                 ####
#----------------------------------------------------------------------------------------------#
#Selección del porcentaje para dividir los datos.
p.tr=0.8;N=nrow(diabetes)
set.seed(223); index.tr=sample(1:N,ceiling(p.tr*N),replace=FALSE)
diabetes.tr=diabetes[index.tr,]         # Datos de entrenamiento del modelo
diabetes.te=diabetes[-index.tr,]        # Datos de prueba para evaluar el performance del modelo

# Función para calcular las métricas de desempeño Rsquared e ICC
perf.metr=function(pred,obs,name){
  ICC = epiR::epi.ccc(pred,obs)
  Rsquared=1- sum((obs-pred)^2)/sum((obs-mean(obs))^2)
  rango=range(c(obs,pred))
  windows()
  plot(obs,pred, pch=20,xlab="Valores Observados",ylab="Valores Predichos", ylim=rango,xlim=rango, main=name)
  abline(a=0,b=1, col="red",lty=2,lwd=2)
  salida = cbind(Rsquared,ICC[[1]][1])
  names(salida)=c("Rsquared","ICC")
  return(round(salida,3))
}

# Se define una estructura fija para todos los modelos 
# garantizando que se emplearán siempre los mismos predictores iniciales.
formula = as.formula(Y~.)

# Se crea una lista para almacenar las medidas de desempeño de los modelos.
Sum.Performance.tr=list()

#----------------------------------------------------------------------------------------------#
# 3.1  Modelo de Regresión Lineal Múltiple - Selección de variables                         ####
#----------------------------------------------------------------------------------------------#
# Modelo saturado (inicial), es decir, modelo con todas las variables.
Sat.model = lm(formula,diabetes.tr)  
summary(Sat.model)

# Identificar las variables que presentan multicolinealidad
car::vif(Sat.model)
windows()
barplot(sort(car::vif(Sat.model),decreasing=TRUE))

# Selección Automática de Variables
Null.model = as.formula(Y~1)    # Definimos el modelo mas sencillo que quisieramos obtener

# Selección de  variables (Backward)
Model_lm = step(Sat.model, direction='backward', scope=Null.model) 
summary(Model_lm)
car::vif(Model_lm)   
windows()
barplot(sort(car::vif(Model_lm),decreasing=TRUE))

# Guardamos los indicadores de MRL en el cuadro de comparación de desempeño
Sum.Performance.tr$MRL=perf.metr(pred=predict(Model_lm),obs=diabetes.tr$Y,name="MRL")

#----------------------------------------------------------------------------------------------#
# 3.2  PLS1                                                                                 ####
#----------------------------------------------------------------------------------------------#                                
# Creación del modelo con un número máximo de componentes 8 y con validación cruzada simple.
Model_pls <- pls::plsr(formula, ncomp=8,scale=TRUE, data=diabetes.tr, validation="CV")

# Se determina el número de componentes a seleccionar para evitar presencia de multicolinealidad.
windows(height=10,width=15)
plot(RMSEP(Model_pls), legendpos = "topright",
     main="Raiz del error para el modelo PLS1")

summary(Model_pls) # 2 Componentes explican el 50.12% de la variabilidad de Y, y
                   # 6 componentes explican el 84.18% de la varianza en X.


Model_pls["coefficients"]   # Se obtienen los coeficientes de todas las componentes principales.
coef(Model_pls,ncomp=2)     # Se obtienen los coeficientes de 2 componentes principales.

# Gráfica del comportamiento de los coeficientes en función de 2 componentes.
windows(height=10,width=15)
plot(Model_pls,plottype='coef',ncomp=1:2,legendpos="bottomleft",labels=names(diabetes.tr))

# Gráfico de barras que muestra la importancia relativa de las variables predictoras en el modelo PLS1. 
windows(height=10,width=15)
barplot(varImp(Model_pls)[,1],names.arg = names(diabetes.tr)[-11])

# Gráfico de correlaciones entre las variables predictoras y las 2 componentes principales del modelo PLS1.
windows(height=10,width=15)
plot(Model_pls,plottype="correlation", labels=names(diabetes.tr))

# Guardamos los indicadores de PLS en el cuadro de comparación de desempeño
Sum.Performance.tr$PLS=perf.metr(pred=predict(Model_pls,ncomp=2)[,1,1],obs=diabetes.tr$Y,name="PLS")

#----------------------------------------------------------------------------------------------#
# 3.3  Arból de regresión                                                                   ####
#----------------------------------------------------------------------------------------------#                                
# Ajuste de un modelo de árbol de decisión
set.seed(223)
Model_tree = rpart::rpart(formula, data=diabetes.tr,control = rpart.control(cp = 0.0001))
printcp(Model_tree)   # Evalúa el número de nodos adecuado para el árbol.

# Se obtiene el valor óptimo de cp
bestcp = Model_tree$cptable[which.min(Model_tree$cptable[,"xerror"]),"CP"]

# Visualización del Árbol
windows(height=10,width=15)
prp(Model_tree, faclen = 0, cex = 0.8, extra = 1)

# Podar el árbol con el cp escogido
Model_tree = rpart::rpart(formula, data=diabetes.tr,control = rpart.control(cp = 0.02135193))

# Visualización del Árbol actualizado por el criterio de parada
windows(height=10,width=15)
prp(Model_tree, faclen = 0, cex = 0.8, extra = 1)

# Guardamos los indicadores de tree en el cuadro de comparación de desempeño
Sum.Performance.tr$tree=perf.metr(pred=predict(Model_tree),obs=diabetes.tr$Y,name="Tree")

#----------------------------------------------------------------------------------------------#
# 3.4  KNN                                                                                  ####
#----------------------------------------------------------------------------------------------#                                
caret::modelLookup('knn') # Se utiliza para realizar validación cruzada y tuneo de hiperparametros

# Se define la estructura de la validación cruzada.
ctrl<- trainControl(method = "repeatedcv",
                    number=10,repeats = 10) # K-fold repetido 10 veces.
valor.k=expand.grid(k=1:10)                 # Malla de valores de exploración para k

# Se ejecuta una validación con tuning, k=1:10
# NOTA: Se exploro k=1:15 y se decidio que el mejor k=10
Model_knn=train(formula, data=diabetes.tr,method="knn",trControl=ctrl,tuneGrid=valor.k)  

# Visualización del mejor K para estimar Y.
windows()
plot(Model_knn,metric="Rsquared")

# Guardamos los indicadores de KNN en el cuadro de comparación de desempeño
Sum.Performance.tr$knn= perf.metr(pred=predict(Model_knn),obs=diabetes.tr$Y,name="KNN")

#----------------------------------------------------------------------------------------------#
# 3.5  SVM                                                                                  ####
#----------------------------------------------------------------------------------------------#
# Ajuste del modelo SVM
tune_svm = e1071::tune(svm,  train.x = diabetes.tr[,-c(2,11)], 
                       train.y = diabetes.tr[,11], 
                       ranges = list(epsilon = seq(0.1,0.5,0.05), cost = 2^(0:5)))
summary(tune_svm)

# Visualización de los resultados del modelo SVM por los diferentes valores de los hiperparámetros
windows(height=10,width=15)
plot(tune_svm)

# Selección del mejor modelo encontrado por la combinación de hiperparámetros epsilon=0.4 y cost=1
Model_svm = tune_svm$best.model
summary(Model_svm)

# Guardamos los indicadores de SVM en el cuadro de comparación de desempeño
Sum.Performance.tr$svm= perf.metr(pred=predict(Model_svm),obs=diabetes.tr$Y,name="SVM")

#----------------------------------------------------------------------------------------------#
# 3.6  Random Forest                                                                        ####
#----------------------------------------------------------------------------------------------#
set.seed(223)
# Sintoniza mtry en el bosque aleatorio por validación cruzada.
# mtry (Número de variables predictoras para el proceso de aleatorización)
caret::modelLookup('rf')

valor.mtry=expand.grid(mtry=2:6)          # malla de exploración para mtry = 2:6

Model_RF<-train(formula ,data=diabetes.tr,method="rf",
                trControl=ctrl,           # K-fold repetido 10 veces.
                tuneGrid=valor.mtry)

windows(height=10,width=15)
plot(Model_RF,metric="Rsquared")

# Bosque aleatorio con m=3 variables predictoras
m=3
n.T=500  # Número de Arboles en el bosque.

#Modelo RF tuneado
Model_RF = randomForest::randomForest(formula,data=diabetes.tr,ntree=n.T,mtry=m,importance=T)

# Visualización de la importancia de las variables.
importancia=importance(Model_RF)
windows(height=10,width=15)
par(mfrow=c(1,2))
barplot(sort(importancia[,2],decreasing=F),col="blue",horiz=T,main=colnames(importancia)[2])
barplot(sort(importancia[,1],decreasing=F),col="blue",horiz=T,main=colnames(importancia)[1])

# Guardamos los indicadores de RF en el cuadro de comparación de desempeño
Sum.Performance.tr$RF= perf.metr(pred=predict(Model_RF),obs=diabetes.tr$Y,name="RF")

#----------------------------------------------------------------------------------------------#
# 3.7  Comparación bondad ajuste                                                            ####
#----------------------------------------------------------------------------------------------#
Sum.Performance.tr = do.call(rbind,Sum.Performance.tr)
View(Sum.Performance.tr)

#----------------------------------------------------------------------------------------------#
## 4. Evaluación de los modelos sobre los datos diabetes-Test                               ####
#----------------------------------------------------------------------------------------------#

Pred=list()

Pred$lm=predict(Model_lm,newdata=diabetes.te)
Pred$pls=predict(Model_pls,newdata=diabetes.te,ncomp=2)
Pred$tree=predict(Model_tree,newdata=diabetes.te)
Pred$knn=predict(Model_knn,newdata=diabetes.te)
Pred$svm=predict(Model_svm,newdata=diabetes.te[,-c(2,11)])
Pred$RF=predict(Model_RF,newdata=diabetes.te)

# Función para calcular las métricas de desempeño ICC y Rsquared
perf.metr.list=function(i,pred,obs,names){
  names.modelo=names[i]
  pred=as.numeric(pred[[i]]);obs=as.numeric(obs)
  ICC = epiR::epi.ccc(pred,obs)
  Rsquared=1- sum((obs-pred)^2)/sum((obs-mean(obs))^2)
  rango=range(c(obs,pred))
  plot(obs,pred, pch=20,xlab="Valores Observados",ylab="Valores Predichos", main=names.modelo,ylim=rango,xlim=rango)
  abline(a=0,b=1, col="red",lty=2,lwd=2)
  salida = cbind(Rsquared,ICC[[1]][1])
  names(salida)=c("Rsquared","ICC")
  return(round(salida,3))
}

windows(height=10,width=15)
par(mfrow=c(2,3))
Sum.performance=data.frame(sapply(seq_along(Pred),perf.metr.list,names=names(Pred),pred=Pred,obs=diabetes.te$Y))
colnames(Sum.performance)=names(Pred)
View(Sum.performance)

