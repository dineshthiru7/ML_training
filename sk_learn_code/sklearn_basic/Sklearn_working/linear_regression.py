import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
np.random.seed(42)
student_df=pd.read_csv(r"D:\learingAndDev\Own_application\Gen_AI_Learnings\Learinings\ML_BASICS\ML_training\Data_science_library\data_sets\student_performance_5000_records.csv")
X_features=student_df[["Sleep_Hours","Math_Score","Attendance_Percentage","Math_Score","Science_Score","Final_Score"]]
X_feature_Status=student_df[["Status"]]
Y_feature=student_df["Grade"]


##preprocession the data 
sim=SimpleImputer(strategy="mean")
X_feature_scale=sim.fit_transform(X_features)

from sklearn.preprocessing import LabelEncoder
lae=LabelEncoder()
Y_feature_scale=lae.fit_transform(Y_feature)

X_feature_Status_scale=lae.fit_transform(X_feature_Status)


print(X_feature_scale[0:5])
print("----")
print(X_feature_Status_scale[0:5])

print("---")
print(Y_feature_scale[0:5])

X_feat_scale_all = np.hstack([X_feature_scale, X_feature_Status_scale.reshape(-1, 1)])
#X_feat_scale_all=np.hstack([X_feature_scale,X_feature_Status_scale])
Y_feature_all=Y_feature_scale

line_model=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(X_feat_scale_all,Y_feature_all)

line_model.fit(x_train,y_train)
pred=line_model.predict(x_test)
print("predection ",pred)
print("score ",line_model.score(x_test,y_test))