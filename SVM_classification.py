import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris = datasets.load_iris()
x=iris.data
#print(iris.feature_names)
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=5) #random_state is a number generator,test_size is the proportion of test data size and total data size
model=svm.SVC(kernel='linear')
model.fit(x_train,y_train)
accuracy=model.score(x_test,y_test)
#print(x_test, y_test)
#pred=model.predict(np.array([4.7, 3.2, 1.3, 0.2]).reshape(1,-1)) #prediction for given input
print(accuracy)
