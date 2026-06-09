## cross val score , kfold , randomKfold,gridsearchcv ,randomgridsearchcv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.neighbors import KNeighborsTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder,MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

np.random.seed(42)
house_price_df=pd.read_csv(r"D:\learingAndDev\Own_application\Gen_AI_Learnings\Learinings\ML_BASICS\ML_training\Data_science_library\data_sets\house_prices_realistic_1000.csv")
X_feature=house_price_df[["sqft","renovated","bedrooms","floors","year_built","lot_size","garage","school_rating","crime_rate","zip"]]
y_output=house_price_df["price"]

data_cleaning_column=["sqft","renovated","bedrooms","floors","year_built","lot_size","garage","school_rating","crime_rate","zip"]
data_clean_stnd_columns=["sqft","lot_size","crime_rate","zip"]
## numerical              

impu_pipe=Pipeline([
    ("imputer",SimpleImputer(strategy="mean"))
])
scaler_imput_pipe=Pipeline([
   ("imputer",SimpleImputer(strategy="mean")),
    ("scaler",StandardScaler())
])

from sklearn.compose import ColumnTransformer 
preProcessing=ColumnTransformer(transformers=[
    ("sccale",scaler_imput_pipe,data_cleaning_column)
])

full_pipeline=Pipeline([
("preProcessor",preProcessing),
 ("model",SGDRegressor(max_iter=1000,tol=1e-3, eta0=0.01, learning_rate='invscaling',
                           penalty='l1',alpha=0.0001,random_state=42))
#   ("model", SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='invscaling',
#                             penalty='l2', alpha=0.0001, random_state=42))


                          ]
                          
                          )
x_train,x_test,y_train,y_test=train_test_split(X_feature,y_output,test_size=0.3,random_state=42)

full_pipeline.fit(x_train,y_train)

score=full_pipeline.score(x_test,y_test)
print("score ",score)

param_dist = {
    "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
    "model__eta0": [1e-3, 1e-2, 1e-1],
    "model__learning_rate": ['constant', 'invscaling', 'adaptive'],
    "model__max_iter": [500, 1000, 2000]
}

from sklearn.model_selection import RandomizedSearchCV
rand_search = RandomizedSearchCV(
    estimator=full_pipeline,
    param_distributions=param_dist,
    n_iter=20,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# fit search instead of calling full_pipeline.fit(...)
rand_search.fit(x_train, y_train)

print('best params:', rand_search.best_params_)
print('best CV score:', rand_search.best_score_)
print('test score:', rand_search.best_estimator_.score(x_test, y_test))