import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tkinter import *
from tkinter import messagebox
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
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

y_pred = model.predict(X_test,verbose=1)
y_pred
CM = metrics.confusion_matrix(y_test,y_pred.round())
print("Confusin Matrix is : \n", CM)
print("#####################################################")
ACC = metrics.accuracy_score(y_test,y_pred.round())
print("accuracy is : \n", ACC)
print("#####################################################")
REC = metrics.recall_score(y_test,y_pred.round())
print("recall is : \n", REC)
print("#####################################################")
Prec = metrics.precision_score(y_test,y_pred.round())
print("precision is : \n", Prec)
print("#####################################################")
F1 = metrics.f1_score(y_test,y_pred.round())
print("f1 score is : \n", F1)
print("#####################################################")
cm = confusion_matrix(y_test, y_pred.round())
plt.figure(figsize=(8, 6))
sns.set(font_scale=1.4)  
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Class 0", "Class 1"]).plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
fpr, tpr, thresholds = roc_curve(y_test, y_pred.round())
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
root = Tk()
root.title("my title")
root.config(bg = "lightblue")
frame = Frame(root)
frame.config(bg = "lightblue")
frame.pack()
employee_info_frame = LabelFrame(frame,text="Employee information")
employee_info_frame.grid(row=0,column=0,padx=20,pady=20)
employee_info_frame.config(bg = "lightblue")
education_label = Label(employee_info_frame,text=" Education",borderwidth=3,width=16)
education_label.grid(row=0,column=0)
JoiningYear_label = Label(employee_info_frame,text="JoiningYear",borderwidth=3,width=16)
JoiningYear_label.grid(row=0,column=1)
education = Entry(employee_info_frame)
JoiningYear = Entry(employee_info_frame)
education.grid(row=1,column=0)
JoiningYear.grid(row=1,column=1)

City_label = Label(employee_info_frame,text="City",borderwidth=3,width=16)
City_label.grid(row=0,column=2)
City= Entry(employee_info_frame)
City.grid(row=1,column=2)

PaymentTier_label = Label(employee_info_frame,text="PaymentTier",borderwidth=3,width=16)
PaymentTier_label.grid(row=2,column=0)
PaymentTier = Entry(employee_info_frame)
PaymentTier.grid(row=3,column=0)

Age_label = Label(employee_info_frame,text="Age",borderwidth=3,width=16)
Age_label.grid(row=2,column=1)
Age = Entry(employee_info_frame)
Age.grid(row=3,column=1)

Gender_label = Label(employee_info_frame,text="Gender",borderwidth=3,width=16)
Gender_label.grid(row=2,column=2)
Gender = Entry(employee_info_frame)
Gender.grid(row=3,column=2)

EverBenched_label = Label(employee_info_frame,text="EverBenched",borderwidth=3,width=16)
EverBenched_label.grid(row=4,column=0)
EverBenched = Entry(employee_info_frame)
EverBenched.grid(row=5,column=0)

ExperienceInCurrentDomain_label = Label(employee_info_frame,text="ExperienceInCurrentDomain",borderwidth=3,width=22)
ExperienceInCurrentDomain_label.grid(row=4,column=1)
ExperienceInCurrentDomain = Entry(employee_info_frame)
ExperienceInCurrentDomain.grid(row=5,column=1)

for widget in employee_info_frame.winfo_children():
    widget.grid_configure(padx=10,pady=5)
button = Button(frame)
def predict():
        data_list = [
        [education.get(),int(JoiningYear.get()), City.get(),int(PaymentTier.get()),int(Age.get()),Gender.get(),EverBenched.get(),\
        int(ExperienceInCurrentDomain.get())]
        ]
        if data_list[0][0] == "Bachelors":
           data_list[0][0] =0 
        elif data_list[0][0] == "Masters":
           data_list[0][0] =1 
        else:
           data_list[0][0] =2   

        mean = np.mean(data["JoiningYear"])
        std = np.std(data["JoiningYear"])
        data_list[0][1] = (data_list[0][1] - mean) / std

        if data_list[0][2] == "Bangalore":
           data_list[0][2] =0 
        elif data_list[0][2] == "New Delhi":
           data_list[0][2] =1 
        else:
           data_list[0][2] =2  
            
        mean = np.mean(data["PaymentTier"])
        std = np.std(data["PaymentTier"])
        data_list[0][3] = (data_list[0][3] - mean) / std    

        mean = np.mean(data["Age"])
        std = np.std(data["Age"])
        data_list[0][4] = (data_list[0][4] - mean) / std   
            
        if data_list[0][5] == "Male":
           data_list[0][5] =1 
        else:
           data_list[0][5] =0 
            
        if data_list[0][6] == "Yes":
           data_list[0][6] =1
        else:
           data_list[0][6] =0    
            
        mean = np.mean(data["ExperienceInCurrentDomain"])
        std = np.std(data["ExperienceInCurrentDomain"])
        data_list[0][7] = (data_list[0][7] - mean) / std               
            
        column_names = ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age','Gender','EverBenched','ExperienceInCurrentDomain']
        df = pd.DataFrame(data_list, columns=column_names)    
        y_pred = model.predict(df,verbose=1)
        result_label.config(text=f"Prediction function executed {str(y_pred[0][0])}")
button = Button(frame,text="predict",command=predict)
button.grid(row=3,column=0,sticky="news",padx=20,pady=20)

result_label = Label(frame, text="", fg="green",bg="lightblue")
result_label.grid(row=4, column=0)
