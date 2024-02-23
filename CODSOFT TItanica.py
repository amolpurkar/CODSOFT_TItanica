#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Name Amol Purkar 
#Importing libriries for data load 


# In[49]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


df=pd.read_csv(r"D:\Imarticus\DATA SETS\titanic3.csv")


# In[4]:


print( "Total Numbber of Rows" ,df.shape[0])
print("Total  NUmber of Columns" , df.shape[1])


# Survived: 0 = No, 1 = Yes
# pclass: Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd
# sibsp: # of siblings / spouses aboard the Titanic
# parch: # of parents / children aboard the Titanic
# ticket: Ticket number
# cabin: Cabin number
# embarked: Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton

# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


pd.set_option("display.max_columns" , None)
pd.set_option("display.max_rows",None)
df


# # Data Visualization using Matplotlib and Seaborn packages.

# In[ ]:





# In[8]:


sns.countplot(x=df.survived, data= df)


# In out of total more are dead

# In[9]:


sns.countplot(x=df.survived,hue=df.sex, data= df)


# Male passeger are dead more than women

# In[10]:


sns.countplot(x=df.survived,hue=df.pclass, data= df);


# survival rate of class one passenger are more than other classes

# In[55]:


sns.heatmap(df.corr(),annot=True)
plt.show()
plt.figure(figsize=(20,50))


# It shows ticket class is impact on Survived feature

# # Feature Enginnering

# In[ ]:





# # MIssing Values

# In[11]:


df.isnull().sum()


# In[12]:


#Roughly 20 % of age data is missing


# In[13]:


sns.heatmap(df.isnull())


# In[14]:


#missing values filling


# In[15]:


df.info()


# In[16]:


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


# In[17]:


#Roughly 20 % of age data is missing


# In[18]:


# delete unnecessary feature from dataset
df =df.drop(["home.dest"], axis =1)
df=df.drop(["name"],axis=1)


# In[19]:


# converting objective data into numeric


# In[20]:


from sklearn.preprocessing import LabelEncoder


# In[21]:


le=LabelEncoder()


# In[22]:


df.info()


# In[23]:


df.sex=le.fit_transform(df.sex)
df.ticket=le.fit_transform(df.ticket)
df.cabin=le.fit_transform(df.cabin)
df.embarked=le.fit_transform(df.embarked)
df.boat=le.fit_transform(df.boat)


# # Model Building

# In[24]:


from sklearn.model_selection import train_test_split


# In[25]:


train, test=train_test_split(df,test_size=0.2)


# In[26]:


train_x=train.drop(["survived"], axis=1)
train_y=train.survived


# In[27]:


test_x=test.drop(["survived"], axis=1)
test_y=test.survived


# In[28]:


from sklearn.linear_model import LogisticRegression


# In[29]:


log=LogisticRegression()


# In[30]:


log.fit(train_x,train_y)



# In[31]:


pred=log.predict(test_x)


# In[32]:


from sklearn.metrics import confusion_matrix


# In[33]:


tab=confusion_matrix(test_y,pred)
tab


# In[34]:


from sklearn.metrics import accuracy_score

accuracy_score(test_y,pred)*100


# In[35]:


from sklearn.svm import SVC
svc=SVC()


# In[36]:


svc.fit(train_x,train_y)


# In[37]:


pred_svc=svc.predict(test_x)


# In[38]:


svc_tab=confusion_matrix(test_y,pred_svc)
svc_tab


# In[39]:


accuracy_score(test_y,pred_svc)*100


# In[40]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines'],
    'Score': [96.66666666666667,96.66666666666667]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head()


# In[ ]:





# In[ ]:





# In[ ]:




