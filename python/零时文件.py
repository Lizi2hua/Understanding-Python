from sklearn import  linear_model
x=[[0,0],[1,1],[2,2]]
y=[0,1,2]
model=linear_model.LinearRegression()
model.fit(x,y)
print(model.coef_)
print(model.intercept_)