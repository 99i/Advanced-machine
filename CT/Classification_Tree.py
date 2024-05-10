import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score, roc_curve, auc


# ['Education', 'JoiningYear', 'City', 'PaymentTier', 'Age', 'Gender',
# 'EverBenched', 'ExperienceInCurrentDomain', 'LeaveOrNot']
df = pd.read_csv("Employee.csv")


# ----------- Converting the categorical values into numbers ----------- #
cat_cols = ['Education', 'City', 'Gender', 'EverBenched']
encoding_mapping = {}
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])
    encoding_mapping[col] = dict(zip(le.classes_, le.transform(le.classes_)))


# ----------- Splitting the columns ----------- #
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ----------- Splitting the data ----------- #
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=76)


# ----------- Building the model ----------- #
alpha = 0.00086
cl_tree = DecisionTreeClassifier(ccp_alpha=alpha)
cl_tree = cl_tree.fit(X_train, y_train)


# ----------- Predicating a row ----------- #
def predict_by_clt_model(info):
    info['JoiningYear'] = np.int64(info['JoiningYear'])
    info['PaymentTier'] = np.int64(info['PaymentTier'])
    info['Age'] = np.int64(info['Age'])
    info['ExperienceInCurrentDomain'] = np.int64(info['ExperienceInCurrentDomain'])

    info['Education'] = np.int64(encoding_mapping['Education'][info['Education']])
    info['City'] = np.int64(encoding_mapping['City'][info['City']])
    info['Gender'] = np.int64(encoding_mapping['Gender'][info['Gender']])
    info['EverBenched'] = np.int64(encoding_mapping['EverBenched'][info['EverBenched']])

    prd_row = pd.DataFrame([info])
    return cl_tree.predict(prd_row)[0]

# # ----------- Calculating the score ----------- #
# y_prd = cl_tree.predict(X_test)
# score_train = cl_tree.score(X_train, y_train)
# score_test = cl_tree.score(X_test, y_test)
# print(f"Training score: {round(score_train, 2)}")
# print(f"Testing score: {round(score_test, 2)}")
#
#
# # ----------- Getting the accuracies from the confusion matrix ----------- #
# acc_sc = accuracy_score(y_test, y_prd)
# f1_sc = f1_score(y_test, y_prd)
# print("-" * 60)
# print(f"The accuracy score: {round(acc_sc, 2)}")
# print(f"The f1-score: {round(f1_sc, 2)}")


# # ----------- Getting the confusion matrix ----------- #
# con_matrix = confusion_matrix(y_test, y_prd, labels=cl_tree.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=con_matrix, display_labels=cl_tree.classes_)
# disp.plot()
# plt.show()


# # ----------- Getting the ROC curve & AUC ----------- #
# rs = cl_tree.predict_proba(X_test)[:, 0]
# fpr, tpr, threshold = roc_curve(y_test, y_prd)
# roc_auc = auc(fpr, tpr)
#
# plt.title('ROC Curve')
# plt.plot(fpr, tpr, color='orange', label=f'AUC = {round(roc_auc, 3)}')
# plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.legend(loc='lower right')
# plt.show()
