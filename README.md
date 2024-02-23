# CODSOFT_TItanica
Intership_Project

contains the details of a subset of the passengers on board (891 passengers

#Name Amol Purkar 
#Importing libriries for data load 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r"D:\Imarticus\DATA SETS\titanic3.csv")

print( "Total Numbber of Rows" ,df.shape[0])
print("Total  NUmber of Columns" , df.shape[1])

Survived: 0 = No, 1 = Yes
pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
sibsp: # of siblings / spouses aboard the Titanic
parch: # of parents / children aboard the Titanic
ticket: Ticket number
cabin: Cabin number
embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

df.describe()

df.info()



pd.set_option("display.max_columns" , None)
pd.set_option("display.max_rows",None)
df

# Data Visualization using Matplotlib and Seaborn packages.



sns.countplot(x=df.survived, data= df)

In out of total more are dead

sns.countplot(x=df.survived,hue=df.sex, data= df)

Male passeger are dead more than women

sns.countplot(x=df.survived,hue=df.pclass, data= df);

survival rate of class one passenger are more than other classes

sns.heatmap(df.corr(),annot=True)
plt.show()
plt.figure(figsize=(20,50))

It shows ticket class is impact on Survived feature

# Feature Enginnering



# MIssing Values

df.isnull().sum()

#Roughly 20 % of age data is missing

sns.heatmap(df.isnull())

#missing values filling

df.info()

df.pclass=df.pclass.fillna(3)
df.survived=df.survived.fillna(0.0)
df.sex=df.sex.fillna("male")
df.age=df.age.fillna(24)
df.sibsp=df.sibsp.fillna(0.0)
df.parch=df.parch.fillna(0.0)
df.ticket=df.ticket.fillna("CA. 2343")
df.fare=df.fare.fillna(df.fare.mean())
df.cabin=df.cabin.fillna("C23 C25 C27")
df.embarked=df.embarked.fillna("S")
df.boat=df.boat.fillna("13")
df.body=df.body.fillna(df.body.mean())

#Roughly 20 % of age data is missing


# delete unnecessary feature from dataset
df =df.drop(["home.dest"], axis =1)
df=df.drop(["name"],axis=1)

# converting objective data into numeric

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

df.info()

df.sex=le.fit_transform(df.sex)
df.ticket=le.fit_transform(df.ticket)
df.cabin=le.fit_transform(df.cabin)
df.embarked=le.fit_transform(df.embarked)
df.boat=le.fit_transform(df.boat)

# Model Building

from sklearn.model_selection import train_test_split

train, test=train_test_split(df,test_size=0.2)

train_x=train.drop(["survived"], axis=1)
train_y=train.survived

test_x=test.drop(["survived"], axis=1)
test_y=test.survived

from sklearn.linear_model import LogisticRegression

log=LogisticRegression()

log.fit(train_x,train_y)



pred=log.predict(test_x)

from sklearn.metrics import confusion_matrix

tab=confusion_matrix(test_y,pred)
tab

from sklearn.metrics import accuracy_score

accuracy_score(test_y,pred)*100

from sklearn.svm import SVC
svc=SVC()

svc.fit(train_x,train_y)


pred_svc=svc.predict(test_x)

svc_tab=confusion_matrix(test_y,pred_svc)
svc_tab

accuracy_score(test_y,pred_svc)*100


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines'],
    'Score': [96.66666666666667,96.66666666666667]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head()





