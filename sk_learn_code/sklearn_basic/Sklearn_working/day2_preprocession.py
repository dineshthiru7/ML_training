import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer


data = {
"age": [20, np.nan, 22, 21, np.nan, 23],
"city": ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai", "Chennai"],
"score": [75, 82, np.nan, 68, 90, 71],
"result": ["pass", "pass", "fail", "fail", "pass", "pass"]
}

student_data=pd.DataFrame(data)
X_feature=student_data[["age","city","score","result"]]
Y_output=student_data["result"]

## result encoding 
lec=LabelEncoder()
Y_scaled=lec.fit_transform(Y_output)
# print(Y_scaled)

##preprocession numeric
age_score=student_data[["age","score"]]
simpt=SimpleImputer(strategy="mean")
X_age_score_scale=simpt.fit_transform(age_score)
# print(X_age_score_scale)

##preprocessing for non numeric
ohe=OneHotEncoder(sparse_output=False)
city_data=student_data[["city"]]
X_city_scale=ohe.fit_transform(city_data)
# print(X_city_scale)

X_train_feature_scale=np.hstack([X_age_score_scale,X_city_scale])
Y_train_scale=Y_scaled


from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X_train_feature_scale,Y_train_scale,test_size=0.3,random_state=42)
linear_mo=LogisticRegression()
model_op=linear_mo.fit(X_train,Y_train)

print(model_op.coef_ , model_op.coef_)
pred=linear_mo.predict(X_test)
print("output : ",pred)

score=linear_mo.score(X_test,Y_test)
print("Score ", score)