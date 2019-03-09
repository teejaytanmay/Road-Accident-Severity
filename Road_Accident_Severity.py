
# coding: utf-8

# In[70]:


import numpy as np
import pandas as pd

#Visualisation Libraries
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from pandas.plotting import scatter_matrix

#Training and Preprocessing Libraries
from xgboost import XGBClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import pickle


# In[71]:


class_names = ['Fatal', 'Severe', 'Slight']


# In[72]:


data= pd.read_csv("accidentva.csv")


# In[73]:


def max_val(s):
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]

year_wise_casualties = data.groupby(['Year'])['Number_of_Casualties'].sum()
year_wise_casualties = year_wise_casualties.reset_index()
year_wise_casualties = year_wise_casualties.style.apply(max_val, axis=0)
year_wise_casualties


# In[74]:


cas_table = data.groupby(['Day_of_Week']).agg({'Number_of_Casualties':['sum'],'Speed_limit':['min','max']})
cas_table = cas_table.sort_values([('Number_of_Casualties','sum')],ascending=False)
cas_table = cas_table.reset_index()
cas_table.style.apply(max_val)


# In[75]:


corr_matrix = data.corr()
corr_matrix["Accident_Severity"].sort_values(ascending=False)


# In[76]:


data.hist(bins=50, figsize=(20,15))
plt.show()


# In[77]:


fig = data.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.6,
                   figsize=(18,11),c="Accident_Severity", cmap=plt.get_cmap("inferno"), 
                   colorbar=True,)


# In[78]:


attributes = ["Number_of_Vehicles","Number_of_Casualties", "Time", "Road_Type", "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions","Accident_Severity"]
scatter_matrix(data[attributes], figsize=(10, 10))


# In[79]:


def preprocessing(data):
    #Drop useless columns and nan values
    data.drop(['Police_Force', 'Junction_Detail', 'Junction_Control', 'Special_Conditions_at_Site', 'Carriageway_Hazards', 'Did_Police_Officer_Attend_Scene_of_Accident', 'LSOA_of_Accident_Location', 'Local_Authority_(District)', 'Local_Authority_(Highway)'], axis=1, inplace=True)
    data.dropna(inplace=True)
    
    #Drop rows with 'Unknown' values
    data = data[data.Weather_Conditions!='Unknown']
    data = data[data.Road_Type!='Unknown']
    
    #Encode "String" Labels into "Int" Labels for easy training
    le = LabelEncoder()
    data["Pedestrian_Crossing-Physical_Facilities"]= le.fit_transform(data["Pedestrian_Crossing-Physical_Facilities"])
    data["Light_Conditions"]= le.fit_transform(data["Light_Conditions"])
    data["Weather_Conditions"] = le.fit_transform(data["Weather_Conditions"])
    data["Road_Surface_Conditions"] = le.fit_transform(data["Road_Surface_Conditions"])
    data["Pedestrian_Crossing-Human_Control"] = le.fit_transform(data["Pedestrian_Crossing-Human_Control"])
    data["Road_Type"] = le.fit_transform(data["Road_Type"])
    
    #Converting Time into Int for easy training
    data["Time"]= data["Time"].astype(str)
    data['Time']=data['Time'].str.slice(0,2, 1)
    data["Time"]= data["Time"].astype(int)
    
    #Creating 3 additional columns, one each for each class we need to classify into
    onehot = pd.get_dummies(data.Accident_Severity,prefix=['Severity'])
    data["Fatal"] = onehot["['Severity']_1"]
    data["Severe"] = onehot["['Severity']_2"]
    data["Slight"] = onehot["['Severity']_3"]
    
    #Finally splitting the data into train and test
    train,test = train_test_split(data,test_size=.25)
    
    return (train,test)


# In[80]:


train,test = preprocessing(data)


# In[81]:


train_features = train[["Number_of_Vehicles","Number_of_Casualties", "Day_of_Week", "Time", "Road_Type", "Speed_limit", "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions","Year", "Urban_or_Rural_Area"]]
test_features =test[["Number_of_Vehicles","Number_of_Casualties", "Day_of_Week", "Time", "Road_Type", "Speed_limit", "Pedestrian_Crossing-Human_Control", "Pedestrian_Crossing-Physical_Facilities", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions","Year", "Urban_or_Rural_Area"]]


# In[ ]:


def model():
    scores = []
    acc_score=[]
    fat_weights = [0.3 for i in range(train["Fatal"].shape[0])]
    sev_weights = [0.5 for i in range(train["Severe"].shape[0])]
    sli_weights = [1 for i in range(train["Slight"].shape[0])]
    class_weights={"Fatal":fat_weights,"Severe":sev_weights,"Slight":sli_weights}
    submission = pd.DataFrame.from_dict({'Accident_Index': test['Accident_Index']})
    for class_name in class_names:
        train_target = train[class_name]
        classifier = EasyEnsembleClassifier(n_estimators=12, base_estimator=XGBClassifier(max_depth=4, learning_rate=0.2, n_estimators=600, silent=True,
                        subsample = 0.8,
                        gamma=0.5,
                        min_child_weight=10,
                        objective='binary:logistic',
                        colsample_bytree = 0.6,
                        max_delta_step = 1,
                        nthreads=1,
                        n_jobs=1))

        cv_score = np.mean(cross_val_score(
            classifier, train_features, train_target, cv=3, scoring='roc_auc'))
        scores.append(cv_score)
#         print('CV score for class {} is {}'.format(class_name, cv_score))

        classifier.fit(train_features, train_target, sample_weight = class_weights[class_name])
        submission[class_name] = classifier.predict_proba(test_features)[:, 1]
        acc = roc_auc_score(test[class_name],submission[class_name])
        acc_score.append(acc)
#         print('Mean accuracy for class {} is {}'.format(class_name,acc))
    
        #Pickling the model
        model_pkl = open('Accident_Severity_Prediction_Model_Pkl.pkl','ab')
        pickle.dump(classifier,model_pkl)
        model_pkl.close()
    
    return (scores,acc_score)
cv,acc = model()


# In[ ]:


print('Total CV score is {}'.format(np.mean(cv)))
print('Total accuracy score is {}'.format(np.mean(acc)))

