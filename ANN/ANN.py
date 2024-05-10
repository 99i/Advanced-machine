import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np

from PIL import ImageTk, Image
data = pd.read_csv("Employee.csv") 
data.dropna(inplace=True)
X = data.drop(columns=["LeaveOrNot"])  # Features
y = data["LeaveOrNot"]
cat_cols = ["Education", "City", "Gender", "EverBenched"]
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
num_cols = ["JoiningYear", "PaymentTier", "Age", "ExperienceInCurrentDomain"]
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.05))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
loss, accuracy = model.evaluate(X_test, y_test)


y_pred = model.predict(X_test,verbose=1)



def predict_ANN(dl2):

    data_list = [[dl2.get("Education"), int(dl2.get("JoiningYear")),dl2.get("City"), int(dl2.get("PaymentTier")), int(dl2.get("Age"))\
                     , dl2.get("Gender"),dl2.get("EverBenched"),int(dl2.get("ExperienceInCurrentDomain"))]]
    if data_list[0][0] == "Bachelors":
        data_list[0][0] = 0
    elif data_list[0][0] == "Masters":
        data_list[0][0] = 1
    else:
        data_list[0][0] = 2

    mean = np.mean(data["JoiningYear"])
    std = np.std(data["JoiningYear"])
    data_list[0][1] = (data_list[0][1] - mean) / std

    if data_list[0][2] == "Bangalore":
        data_list[0][2] = 0
    elif data_list[0][2] == "New Delhi":
        data_list[0][2] = 1
    else:
        data_list[0][2] = 2

    mean = np.mean(data["PaymentTier"])
    std = np.std(data["PaymentTier"])
    data_list[0][3] = (data_list[0][3] - mean) / std

    mean = np.mean(data["Age"])
    std = np.std(data["Age"])
    data_list[0][4] = (data_list[0][4] - mean) / std

    if data_list[0][5] == "Male":
        data_list[0][5] = 1
    else:
        data_list[0][5] = 0

    if data_list[0][6] == "Yes":
        data_list[0][6] = 1
    else:
        data_list[0][6] = 0

    mean = np.mean(data["ExperienceInCurrentDomain"])
    std = np.std(data["ExperienceInCurrentDomain"])
    data_list[0][7] = (data_list[0][7] - mean) / std

    column_names = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender', 'EverBenched',
                    'ExperienceInCurrentDomain']
    df = pd.DataFrame(data_list, columns=column_names)
    y_pred = model.predict(df, verbose=1)
    return np.round(y_pred[0][0])