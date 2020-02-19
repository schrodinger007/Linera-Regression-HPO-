import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
X=pd.read_csv(r'C:\Users\91914\Downloads\Training Data (2)\Linear_X_Train.csv')
y=pd.read_csv(r'C:\Users\91914\Downloads\Training Data (2)\Linear_Y_Train.csv')
X.head()
#y.head()
plt.style.use('seaborn')
plt.xlabel("Hardwork")
plt.ylabel("Performance")
plt.scatter(X,y,color='orange')
plt.show()
X.shape,y.shape
X=X.values
y=y.values
u=X.mean()
std=X.std()
print(u,std)
X=(X-u)/std
def hypothesis(x,theta):
    y_=theta[0]+theta[1]*x
    return y_
def gradient(X,Y,theta):
    m=X.shape[0]
    grad=np.zeros((2,))
    for i in range(m):
        x=X[i]
        y_=hypothesis(x,theta)
        y=Y[i]
        grad[0]+=(y_-y)
        grad[1]+=(y_-y)*x
    return grad/m
def error(X,Y,theta):
    m=X.shape[0]
    total_error=0.0
    for i in range(m):
        y_=hypothesis(X[i],theta)
        total_error+=(y_-Y[i])**2
    return total_error/m
def gradientDescent(X,Y,max_steps=100,learning_rate=0.1):
    theta=np.zeros((2,))
    error_list=[]
    for i in range(max_steps):
        grad=gradient(X,Y,theta)
        e=error(X,Y,theta)
        error_list.append(e)
        theta[0]=theta[0]-learning_rate*grad[0]
        theta[1]=theta[1]-learning_rate*grad[1]
    return theta,error_list
theta,error_list=gradientDescent(X,y)
plt.plot(error_list)
plt.title("Reduction in Error over time")
y_=hypothesis(X,theta)
print(y_)
plt.scatter(X,y)
plt.plot(X,y_,color='orange',label="Prediction")
plt.legend()
plt.show()
X_test=pd.read_csv(r'C:\Users\91914\Downloads\Test Cases\Linear_X_Test.csv').values
y_test=hypothesis(X_test,theta)
y_test.shape
df=pd.DataFrame(data=y_test,columns=["y"])
df.to_csv('y_prediction.csv',index=False)
def r2_score(Y,Y_):
    num=np.sum((Y-Y_)**2)
    denom=np.sum((Y-Y.mean())**2)
    score=(1-num/denom)
    return score*100
r2_score(y,y_)