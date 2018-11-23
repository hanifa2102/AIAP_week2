def plotScatter(X,fX):
    fig=plt.figure()
    ax=fig.add_subplot(111)
    #ax.plot(X,fX,color='red',linewidth=3)
    ax.scatter(X,fX,color='blue',s=3)
    plt.show()
def funcX(x):
    fx=0.2*x**5-3.5*x**4+21*x**3-53*x**2+37*x+0.6
    fx=x**2
    return fx


x=np.arange(0,7,0.005)
fx=funcX(x)
plotScatter(x,fx)
noise=np.random.normal(0,2,x.size)
fx_=fx+noise
plotScatter(x,fx_)

X_train,X_test,y_train,y_test=train_test_split(x,fx_,test_size=300,random_state=42)
plotScatter(X_test,y_test)

lrModelsList=[]
numOfModels=2
for i in range(0,numOfModels):
    index=np.random.randint(low=0,high=X_train.size,size=80)
    lr=LinearRegression()
    lr.fit(X_train[index].reshape(-1,1),y_train[index].reshape(-1,1))
    lrModelsList.append(lr)
    
y_pred_ar=np.zeros((300,numOfModels))
mse_ar=np.zeros((numOfModels,1))
i=0
for lrModel in lrModelsList:
    y_pred=lrModel.predict(X_test.reshape(-1,1))
    y_pred_ar[:,i:i+1]=y_pred
    mse_ar[i]=mean_squared_error(y_test, y_pred)
    i=i+1

y_true=funcX(X_test)
y_pred_mean=y_pred_ar.mean(axis=1).reshape(-1,1)
y_pred_mean_rep=np.repeat(y_pred_mean,numOfModels,axis=1)

