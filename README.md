# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program
2. Import the python pandas library as pd
3. Read the contents of the Spam csv file
4. Display the first 5 rows of the dataset using head()
5. Assign x as v1 values and y as v2 values
6. From sklearn library select the feature extraction and import CountVectorizer
7. CountVectorizer will convert the Text to Numerical Data
8. From sklearn library import Support Vector Classifier (ie. SVC)
9. Predict the x_test using SVC
10. Print the accuracy of the SVM Model
11. Stop the program. 

## Program:
```py
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRIYANKA.A
RegisterNumber: 212222230113
*/
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["EmailText"].values
y=data["Label"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![SVM For Spam Mail Detection](sam.png)
## data.head()
![282037973-3de80162-e0a8-4f20-956a-05a1964caccf](https://github.com/DhanalakshmiCSE/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477832/8a93aa7d-8d47-4d0d-aa67-a11368a5eaad)
## data.info()
![282038065-c61d9e61-51b5-45c3-9320-18ff49052334](https://github.com/DhanalakshmiCSE/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477832/36dc28e5-c985-4c51-92dc-cbad999c2d3d)
## data.isnull() & sum()
![282038266-d538cec1-1ec4-4268-a3d2-3a80ec54b924](https://github.com/DhanalakshmiCSE/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477832/012296ef-98c5-4605-946e-6d325f37e95e)
## Y_prediction:
![282038348-217a9c70-ddbd-492f-9c28-5cd8396aa7aa](https://github.com/DhanalakshmiCSE/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477832/a639f83a-37ca-4d61-b9ce-804f6a697ed0)
## Accuracy:
![282038451-64a74518-ce6f-4153-98e1-d7b58e3dd230](https://github.com/DhanalakshmiCSE/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119477832/cd51dc21-7105-46a9-8222-8da9f209a035)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
