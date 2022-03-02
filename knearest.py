from sklearn.neighbours  import KNeighboursclassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
irisData=load_iris()
x=irisData.data
y=irisData.target
x_train,x_test,y_train,y_test=(train_test_split(x,y,testsize=0.2,random_state=42))
Knn=KNeighboursclassifier(n_neighbour=7)
Knn.fit(x_train,y_train)
print(Knn.predict(x_test))