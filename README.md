# RandomForrestClassifier
How to apply Random Forrest Algorithm to any data set Or How Random Forrest Algo gave me 98% accuracy on Breast Cancer Prediction Data.

---

I am so pleased to see you have reached a stage where you are now predicting accuracy of your models using Model algorithm's, today I'd like to share one of my favourite algorithm's - Random forrest Classifier (or RFC)
Now before we even begin talking about the code or how it is done, I'd like to cover why, what of RFC.

What is it?

It's an algorithm to help your machine learn from training data you provide, and the learning helps in algorithm to predict outcome of data points (as inputs) you do not know of OR to predict the output of data you do not have.
You need to know a little about decision trees to start with RFC. A decision tree looks like this (below) - 
Decision trees start with a question, next step is a simple yes or no classification of it, step 2 is also a simple yes or no classification and voila you have a tree full of choices. You see 

Have I 25$ 

1.1 Go to restaurant 
1.2 Buy a hamburger
2. Go to sleep

1.1, 1.2 and 2 are all called leaf nodes or decisions, for now just keep hold of this concept and I will explain you it's significance later.

Why do we use RFC ?

RFC algo helps us understand what dimensions or features to pick or make a model on since it helps in finalising or narrowing features. 
Now imagine having a csv with 29 columns in it and you need to make a ML model on it, choose RFC to narrow down the most important features/columns to focus on and then make a ML model around it. 


Note : You should consider that the breast cancer data I dealt with had only few hundred data points, if your dataset is large say 200 columns and over thousands of data points - RFC is slow and probably not the best way to go about it.

How do we apply RFC ?

let me break this down into 1,2,3 for you -
Get the data i.e. load the csv file or dataset into a dataframe using either pandas or you can use an exciting dataset like I did.

from sklearn.datasets import load_breast_cancer
d1= load_breast_cancer()
This is not the only way to load data, you can also do - 
import pandas as pd
d1=pd.read_csv('path of File in local drive')
2. Time to make decision trees from all of your columns - This is the tricky bit, before I explain this to you, you need to know about Regression and classification, if you don't read the below para , if you do, skip it.


---

What is regression and classification ?
There are two types of techniques we use to predict output from input - 

Regression
Classification 

Regression is essentially finding the right function which fits all inputs and is able to predict out. Something like below - where the function is able to justify all points, say this function is f(x) where x is the input, 
Regression helps us find the function and then say we have new input points which are not available on this graph, we use f(x) to find output of those. f(x) can be a differential function, trignometric function, a custom function or even a simple y=mx + c (linear function).

Classification is making sets or categories of inputs based on a function or a rule, imagine data as {'Tomato', 'Apple', 'Banana', 'Potato'} as input (x). Then 

X1 = fruits is a category and it shall have {'Apple', 'Banana'} 
X2 = vegetables is a category containing {'Tomato', 'Potato'}
and if a new input say {'Cucumber'} is added to data set, using classification we can place it in X2.


---

Coming back to Time to make decision trees from all of your columns, we now know we use both regression and classification + RFC algo to find output or predict output , however the approach is different for both.

In case of regression, 
after making a Decision tree (DT) of all columns , all leaf's or output or nodes from a single column or dimension of our data are taken, then the average of all data is done to find the Recommendation from that column.

Let me explain this via code >

print(d1.feature_names)
>> ['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean smoothness' 'mean compactness' 'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' 'worst symmetry' 'worst fractal dimension']
 
There are 29 columns, with hundreds of data, we basically take random samples from all columns and make hundreds of decision trees and have hundreds x n leaf's or nodes our of which we find a recommendation value from the column.

In case of classification,
after making a Decision tree (DT) of all columns , all leaf's or output or nodes from a single column or dimension of our data are taken, then voting is done i.e. most favourable result is chosen as recommendation from that column.

Phew, that was tough to grasp, re read it and try and make sense out of it.

Random forests is a supervised learning algorithm. It can be used both for classification and regression. It is also…

3. Last step is to take a vote from all recommendation's from all columns and choosing a value- this is RFC algo. Random forrest is given it's name because we randomly take data from features to make decision trees and decide output post voting amongst recommended values of those trees. Since we consider hundreds of decision trees and then select most favourable or average value for recommendation therefore removing any bias in our recommendation which is a by product of RFC - no bias of data.

#RFC algo implementation using code
from sklearn.model_selection import train_test_split 
#to divide training & testing data
from sklearn.ensemble import RandomForestClassifier
#library to import RFC algo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
#dividing data into 70:30 ratio i.e. 70 for Training and 30 for testing 
clf=RandomForestClassifier(n_estimators=100)
#making 100 decision trees from our data d1
clf.fit(X_train,y_train)
#training our machine by providing x inputs and providing answers as y_train
y_pred=clf.predict(X_test)
#calculating output of data not given to machine as Training data to check if our model works
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


>> My accuracy came out to be 94% [ Remember this will wary based on random samples picked up from machine. ]

What did we learn ?
We learnt what is machine learning ? , we learnt regression and classification techniques and most importantly we learnt how to apply these concepts to code.

If you would like to run the code on your notebook, here is the github link of the same.

However, tutorial is not over.
If you want to increase the accuracy of your prediction there is a further step.
Further step, 

Remember we discussed an accuracy score. It's most critical for next step, let me cover it -

What is accuracy score ?

It's called guinea score or relevance score and is super important for accuracy of our prediction.
Each column has a relevance score which can be found out - 
feature_imp =  pd.Series(clf.feature_importances_,index=d1.feature_names).sort_values(ascending=False)

The sum of all accuracy scores or relevance scores is 1 

>>mean radius 0.164371 i.e. 16.4%
mean perimeter 0.150038 i.e. 15% 
worst radius 0.094000 ….
worst area 0.079068
mean fractal dimension 0.067840
worst perimeter 0.060207
mean smoothness 0.053101
mean area 0.052318
mean symmetry 0.037666
mean concave points 0.029321
worst fractal dimension 0.024776
worst concave points 0.021425
symmetry error 0.019996
worst smoothness 0.018007
mean concavity 0.015951
worst concavity 0.015148
worst symmetry 0.012145
mean texture 0.010561
worst texture 0.009160
worst compactness 0.008927
texture error 0.007539
perimeter error 0.006698
area error 0.006677
concave points error 0.006199
radius error 0.006158
concavity error 0.006010
compactness error 0.005264
smoothness error 0.004375
mean compactness 0.004162
fractal dimension error 0.002891

As you can see Fractal dimension, mean compactness are all <0 % impactful on our algo which implies we can reduce these dimensions or not use them.
It also implies RFC has given us all important dimensions to focus on, which are all dimensions > 1% or 2 % i.e. above Worst texture.

By knowing important dimensions, our focus can shift to adding more input values and training values for important dimensions so that our prediction can improve.

After removing the non useful dimensions, my accuracy score was 96.4%.
Hope you were able to learn from this tutorial, do like and share with you friends on medium or github :)
May the force be with you.
