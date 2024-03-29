# Evolutionary feature selection: a novel wrapper feature selection architecture based on evolutionary strategies
Feature selection algorithm for machine learning models based on evolution 

## Step by step instructions to make it work

First of all the user need to define its own evaluation function for its model in the described format along with its features that the user wants to select


## Setup

To install evfs you can install it directly from the repo with pip:

```pip install git+https://github.com/cloudwalk/simple-evolutionary-feature-search.git```

```python
import evfs 

from evfs import efs 

```

###### the format for evaluation function :

```python
class evalFunction():
    
    def __init__(self,features):
        #defined features in list format  
        self.features=features
    
    #change according to user uses case for model
    def func(self,x_train,x_test,y_train,y_test,gen):
        
        #define your model to judge with features here 
        
        #model defined here is just for an example purpose 

        testModel=AdaBoostClassifier().fit(x_train[self.features],y_train)

        pred=testModel.predict(x_test[self.features])
       
        scores=accuracy_score(y_test,pred)

        #important to return an integer score 
        return scores
```
## How to run evolutionary feature selector 

###### Define the variables for evalFunction

```python

variables=[x_train,x_test,y_train,y_test] 

string="chaos"


"""
generations= number of generations we want the efs to run number of generations < max number of features we want to discover i.e  generations < len(features ) 

features= predefined features variable.

dicName= name with which we want to save the dictionary containing features.

creaturesNumber= total number of random creatures in each generation 

string= if it is "chaos" then efs will use chaos otherwise simple efs ,default value for this variable is "chaos". 
"""
#default values 
start=0

generations=10 
#this means that evfs will run form 0 - 10 generations if you change the  start = 30 & generations=50 it will run from 30 - 50 generations only
# use this type of custom limit in to reduce the runtime of algorithm
 

features=[f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10] 

dicName="efs0"

creaturesNumber=100

testefs=efs.EvolutionaryFeatureSelector(start,generations,features,dicName,creaturesNumber)

#use string if you don't want to use chaos, this will make algorihtm runs faster 

testefs.select_features(variables,evalFunction,string)

```



