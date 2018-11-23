import numpy as np
import matplotlib.pyplot as plt

def func(X):
    fX=0.2*X**5-3.5*X**4+21*X**3-53*X**2+37*X+0.6
    return fX

def plotit(X,fX):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    #ax.plot(X,fX,color='red',linewidth=3)
    ax.scatter(X,fX,color='lightblue',s=3)


y_pred_mean=y_pred.mean(axis=1).reshape(-1,1)
y_pred_mean_rep=np.repeat(y_pred_mean,500,axis=1)

Ex_sq= (y_pred-y_pred_mean_rep)**2