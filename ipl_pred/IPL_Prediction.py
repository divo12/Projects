#!/usr/bin/env python
# coding: utf-8

# In[2]:



import pandas as pd
import pickle
data=pd.read_csv('https://raw.githubusercontent.com/akash2k/IPL-Score-Prediction/main/ipl2017.csv')


# In[3]:


data.head()


# In[4]:


columns_to_remove={'mid','venue','batsman','bowler','striker','non-striker'}
data.drop(labels=columns_to_remove,axis=1, inplace=True)


# In[5]:


data['bat_team'].unique()


# In[6]:


data=data.replace({'Delhi Daredevils':'Delhi Capitals'},regex=True)
data['bat_team'].unique()


# In[7]:


teams_currently=['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals','Mumbai Indians','Kings XI Punjab','Royal Challengers Bangalore','Delhi Capitals', 'Sunrisers Hyderabad']


# In[8]:


#isin used for filtering
data=data[(data['bat_team'].isin(teams_currently)) & (data['bowl_team'].isin(teams_currently))]


# In[9]:


#removing the first 5 overs in the match
data=data[data['overs']>=5.0]


# In[10]:


data.head()


# In[11]:


print(data['bat_team'].unique())


# In[12]:


#converting date to datetime object
from datetime import datetime
data['date'] = data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))


# In[13]:


#apply one-hot encoding for categorical data i.e. team name
encodeddata=pd.get_dummies(data=data,columns=['bat_team','bowl_team'])


# In[14]:


encodeddata.head()


# In[15]:


encodeddata.columns


# In[16]:


#rearranging the columns
encodeddata=encodeddata[['date','bat_team_Chennai Super Kings', 'bat_team_Delhi Capitals',
       'bat_team_Kings XI Punjab', 'bat_team_Kolkata Knight Riders',
       'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
       'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
       'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Capitals',
       'bowl_team_Kings XI Punjab', 'bowl_team_Kolkata Knight Riders',
       'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
       'bowl_team_Royal Challengers Bangalore',
       'bowl_team_Sunrisers Hyderabad','overs', 'runs', 'wickets','runs_last_5', 'wickets_last_5','total']]


# In[17]:


X_train = encodeddata.drop(labels='total', axis=1)[encodeddata['date'].dt.year <= 2016]
X_test = encodeddata.drop(labels='total', axis=1)[encodeddata['date'].dt.year >= 2017]


# In[18]:


y_train = encodeddata[encodeddata['date'].dt.year <= 2016]['total'].values
y_test = encodeddata[encodeddata['date'].dt.year >= 2017]['total'].values


# In[19]:


X_train.drop(labels='date', axis=True,inplace=True)
X_test.drop(labels='date',axis=True,inplace=True)


# In[20]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[21]:


#creating a pickle file
filename='first-innings-score-lr-model.pkl'
pickle.dump(regressor,open(filename,'wb'))


# In[22]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV


# In[23]:


ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=RandomizedSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)


# In[24]:


prediction=ridge_regressor.predict(X_test)


# In[25]:


import seaborn as sns
sns.distplot(y_test-prediction)


# In[26]:


from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# In[27]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV


# In[28]:


lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=RandomizedSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(X_train,y_train)


# In[29]:


prediction1=lasso_regressor.predict(X_test)
from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction1))
print('MSE:', metrics.mean_squared_error(y_test, prediction1))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction1)))


# In[ ]:


from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'first-innings-score-lr-model.pkl'
regressor = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    if request.method == 'POST':
        
        batting_team = request.form['batting-team']
        if batting_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif batting_team == 'Delhi Daredevils':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif batting_team == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif batting_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif batting_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif batting_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif batting_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif batting_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
            
        bowling_team = request.form['bowling-team']
        if bowling_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0]
        elif bowling_team == 'Delhi Daredevils':
            temp_array = temp_array + [0,1,0,0,0,0,0,0]
        elif bowling_team == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0]
        elif bowling_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0]
        elif bowling_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0]
        elif bowling_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0]
        elif bowling_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0]
        elif bowling_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1]
            
            
        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])
        
        temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
        
        data = np.array([temp_array])
        my_prediction = int(regressor.predict(data)[0])
              
        return render_template('result.html', lower_limit = my_prediction-10, upper_limit = my_prediction+5)



if __name__ == '__main__':
	app.run(debug=False)


# In[ ]:




