import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df=pd.read_csv("regression_df.csv")
df=df.fillna(0)

df.describe()
x=df.iloc[:,2:].values
y=df['rating'].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=100)


scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)



model=SVR(kernel='linear')
model_traned=model.fit(x_train_scaled,y_train)
y_predicted = model_traned.predict(x_test_scaled)

r2=r2_score(y_test, y_predicted)
print(f"R^2 Score: {int(r2*100)}%")

mae = mean_absolute_error(y_test, y_predicted)
print("Mean Absolute Error:", mae)


def predect_SVR(ranks):
    rank = np.array(list(ranks.values())).reshape(1, -1)
    return  int(model_traned.predict(scaler.transform(rank)))


plt.scatter(y_test, y_predicted, color = 'darkorange')
plt.plot(y_test, y_test, color = 'cornflowerblue')
plt.title('CodeForces rank (SVR)')
plt.xlabel('Contest points')
plt.ylabel('Rank rate')
plt.show()
