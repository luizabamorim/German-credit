###########################################################################
#
#                   Classificação de crédito 
#                 
###########################################################################
######################Pacotes Utilizados ##################################

install.packages("e1071")  #Naive Bayes e SVC
install.packages("rpart")  #Decision Tree
install.packages("rpart.plot")  #Visualização da árvore
install.packages("randomForest") # Random Forest
install.packages("class")    #KNN
install.packages("caret") #Matriz de Confusão
install.packages("ggplot2") #Gráficos
install.packages("Amelia")  #Missing values
install.packages("dplyr")  #Transformação de dados

######################Carregando Pacotes ##################################
library(e1071) 
library(rpart) 
library(rpart.plot)
library(randomForest)
library(class)
library(caret)
library(ggplot2)
library(Amelia)
library(dplyr)

######################  Dados  ############################################
#Conhecenddo os dados e visualizando sua estrutura
dados=read.csv("German_credit.csv") #Leitura do Banco de Dados
View(dados) #Visualizando os dados
summary(dados)
str(dados)  #Estrutura dos dados

dados %>% select(Duration.of.Credit..month.,Credit.Amount,Age..years.) %>% 
  summary()    #Estatísticas das variáveis contínuas

ggplot(dados,aes(Duration.of.Credit..month.))+
  geom_histogram(fill="dark blue",col="white")+
  labs(subtitle = "Histograma", y = "Frequência",
       x = "Duração mensal de crédito")+
  theme_bw()

ggplot(dados,aes(Age..years.))+
  geom_histogram(fill="dark blue",col="white")+
  labs(subtitle = "Histograma", y = "Frequência",
       x = "Idade")+
  theme_bw()

ggplot(dados,aes(Credit.Amount))+
  geom_histogram(fill="dark blue",col="white")+
  labs(subtitle = "Histograma", y = "Frequência",
       x = "Montante de crédito")+
  theme_bw()


ggplot(dados, aes(Creditability)) +
  geom_bar(color = "white", 
           fill = "dark blue") +
  labs(title = "                         Concessão de Crédito")+
  theme_bw()

#Transformando os dados em fatores


dados$Creditability =as.factor(dados$Creditability) 
dados$Account.Balance=as.factor(dados$Account.Balance)
dados$Payment.Status.of.Previous.Credit=as.factor(dados$Payment.Status.of.Previous.Credit)
dados$Purpose=as.factor(dados$Purpose)
dados$Value.Savings.Stocks=as.factor(dados$Value.Savings.Stocks)
dados$Length.of.current.employment=as.factor(dados$Length.of.current.employment)
dados$Instalment.per.cent=as.factor(dados$Instalment.per.cent)
dados$Sex...Marital.Status=as.factor(dados$Sex...Marital.Status)
dados$Guarantors=as.factor(dados$Guarantors)
dados$Duration.in.Current.address=as.factor(dados$Duration.in.Current.address)
dados$Most.valuable.available.asset=as.factor(dados$Most.valuable.available.asset)
dados$Concurrent.Credits=as.factor(dados$Concurrent.Credits)
dados$Type.of.apartment=as.factor(dados$Type.of.apartment)
dados$No.of.Credits.at.this.Bank=as.factor(dados$No.of.Credits.at.this.Bank)
dados$Occupation=as.factor(dados$Occupation)
dados$No.of.dependents=as.factor(dados$No.of.dependents)
dados$Telephone=as.factor(dados$Telephone)
dados$Foreign.Worker=as.factor(dados$Foreign.Worker)

str(dados)

#Verificando se existem NA

sapply(dados, function(x) sum(is.na(x)))
missmap(dados,main="Valores Faltantes") #Gráfico para verificação dos NA


##Divisão da base de dados entre TREINAMENTO e TESTE (75% para Treinamento e 25% para teste)
#Amostragem estratificada

set.seed(1) #Possibilitar utilizar sempre os mesmos dados
amostra_teste <- createDataPartition(dados$Creditability,p = 0.25,list = FALSE)
teste=dados[amostra_teste,]
treinam=dados[-amostra_teste,]

#Verificando se foi mantida a mesma proporção de 0 e 1 na amostra que será utilizada para teste e para treinamento
prop.table(table(teste$Creditability))
prop.table(table(treinam$Creditability))
prop.table(table(dados$Creditability))

Comp=cbind(prop.table(table(teste$Creditability)),prop.table(table(treinam$Creditability)),prop.table(table(dados$Creditability)))
colnames(Comp)=c("Teste","Treinamento","Dados")
Comp


######################  MODELOS  ############################################


#  1) REGRESSÃO LOGÍSTICA

modelo_rl=glm(Creditability ~ .,family=binomial(link="logit"),data=treinam) #Ajustando o modelo
prev_rl=predict(modelo_rl,type="response",newdata=teste[,2:21])  #Fazendo previsões
prev_rl=ifelse(prev_rl>0.5, 1, 0)  #Previsão
matriz_rl=table(teste[,1],prev_rl)
conf_rl=confusionMatrix(matriz_rl)  #Matriz de confusão
conf_rl
summary(modelo_rl)

par(mfrow=c(2,2))
plot(modelo_rl)



#  2) MODELO NAIVE BAYES

modelo_nb=naiveBayes(treinam[,2:21],treinam[,1]) #Modelo utilizando Naive Bayes
prev_nb=predict(modelo_nb,teste[,2:21]) #Previsão utilizando o modelo
matriz_nb=table(teste$Creditability,prev_nb)  #Matriz para comparação das previsões feitas com o modelo e os valores reais
conf_nb=confusionMatrix(matriz_nb) #Verificação da acurácia do modelo
conf_nb


#   3) Modelo Árvore de Decisão

modelo_ad=rpart(Creditability~.,treinam)  #Gerando uma árvore para classificação
print(modelo_ad)                          #Visualização da árvore
prev_ad=predict(modelo_ad,teste[,2:21],"class") #Previsão utilizando a árvore gerada
rpart.plot(modelo_ad)                         #Visualização gráfica da árvore
matriz_ad=table(teste$Creditability,prev_ad)
conf_ad=confusionMatrix(matriz_ad)    #Matriz de confusão para verificação da acurácia
conf_ad
help(svm)

#  4) Random Forest

modelo_rf=randomForest(Creditability ~.,data=treinam)
print(modelo_rf)
prev_rf=predict(modelo_rf,teste[,2:21])
matriz_rf=table(teste[,1],prev_rf)
conf_rf=confusionMatrix(matriz_rf)
conf_rf
plot(modelo_rf)

#  5) SVM

modelo_svm=svm(formula=Creditability ~ .,data=treinam,type="C-classification",kernel="radial",cost=5)
#modelo_svm=svm(formula=Creditability ~ .,data=treinam,type="C-classification",kernel="sigmoid",cost=5)
#modelo_svm=svm(formula=Creditability ~ .,data=treinam,type="C-classification",kernel="linear",cost=5)
#modelo_svm=svm(formula=Creditability ~ .,data=treinam,type="C-classification",kernel="polynomial",cost=5)
print(modelo_svm)
prev_svm=predict(modelo_svm,teste[,2:21])
matriz_svm=table(teste[,1],prev_svm)
conf_svm=confusionMatrix(matriz_svm)
conf_svm

#  6) KNN

#Escalonamento
dados[,c(3,6,14)]=scale(dados[,c(3,6,14)]) # O algoritmo é baseado em distâncias, logo precisamos colocar os dados em mesma escala

set.seed(1) #Possibilitar utilizar sempre os mesmos dados
amostra_teste <- createDataPartition(dados$Creditability,p = 0.25,list = FALSE)
teste=dados[amostra_teste,]
treinam=dados[-amostra_teste,]

prev_knn=knn(treinam[,2:21],teste[,2:21],treinam[,1],k=5)
matriz_knn=table(teste[,1],prev_knn)
conf_knn=confusionMatrix(matriz_knn)
conf_knn

############ Tentando melhorar os modelos

###### Fazendo algumas modificações no modelo Random Forest
#Importância das variáveis

varImpPlot(modelo_rf,main="Variáveis mais importantes") #Variáveis mais importantes para o modelo
import=importance(modelo_rf)

##Retirando as variáveis consideradas menos importantes pelo modelo
dados$No.of.dependents=NULL
dados$Telephone=NULL
dados$Foreign.Worker=NULL

set.seed(1) #Possibilitar utilizar sempre os mesmos dados
amostra_teste <- createDataPartition(dados$Creditability,p = 0.25,list = FALSE)
teste=dados[amostra_teste,]
treinam=dados[-amostra_teste,]

modelo_rf=randomForest(x=treinam[,2:18],y=treinam$Creditability)
print(modelo_rf)
prev_rf=predict(modelo_rf,teste[,2:18])
matriz_rf=table(teste[,1],prev_rf)
conf_rf=confusionMatrix(matriz_rf)
conf_rf

### Fazendo a mesma coisa para o SVM

modelo_svm=svm(formula=Creditability ~ .,data=treinam,type="C-classification",kernel="radial",cost=5)
modelo_svm=svm(formula=Creditability ~ .,data=treinam,type="C-classification",kernel="sigmoid",cost=5)
print(modelo_svm)
prev_svm=predict(modelo_svm,teste[,2:18])
matriz_svm=table(teste[,1],prev_svm)
conf_svm=confusionMatrix(matriz_svm)
conf_svm

