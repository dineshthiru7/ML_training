import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

np.random.seed(42)
student_df=pd.read_csv(r"D:\learingAndDev\Own_application\Gen_AI_Learnings\Learinings\ML_BASICS\ML_training\Data_science_library\data_sets\house_prices_realistic_1000.csv")

print(student_df.columns.to_list())

X_feature=student_df[["bedrooms","bathrooms","sqft","floors","school_rating"]]
Y_output=student_df[["price"]]

from sklearn.impute import SimpleImputer
imp=SimpleImputer(strategy="mean")
x_eval=imp.fit_transform(X_feature)

from sklearn.model_selection import train_test_split
x_tr,x_tes,y_tr,y_tes=train_test_split(X_feature,Y_output,test_size=0.4,random_state=42)

from sklearn.linear_model import LinearRegression
lin_mod=LinearRegression()
model_tr=lin_mod.fit(x_tr,y_tr)

y_pred=model_tr.predict(y_tes)

score=model_tr.score(x_tes,y_tes)
print(score)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae=mean_absolute_error(y_tes,y_pred)
mse=mean_squared_error(y_tes,y_pred)
r2=r2_score(y_tes,y_pred)

print("mae",mae)
print("mse",mse)
print("r2",r2)
