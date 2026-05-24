import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

##Percentage,Result (Pass / Fail),Grade
std_df=pd.read_csv(f"D:\learingAndDev\Own_application\Gen_AI_Learnings\Learinings\ML_BASICS\ML_training\Data_science_library\data_sets\std.csv")
X_feature=std_df[["Percentage","Grade"]]
Y_output=std_df[["Result (Pass / Fail)"]]

def with_preprocess():
            

        # ##data pre processing

        # from sklearn.impute import SimpleImputer
        # sim_most_freq=SimpleImputer(strategy="most_frequent")
        # X_feature_NAN=sim_most_freq.fit_transform(X_feature)

        from sklearn.preprocessing import StandardScaler,LabelEncoder,OrdinalEncoder
        lbl=LabelEncoder()
        std=StandardScaler()
        ord=OrdinalEncoder()

        grade_hierarchy = ['U', 'D', 'C', 'B', 'A','A+']
        encoder = OrdinalEncoder(categories=[grade_hierarchy])
        x_feature_grade_encode = encoder.fit_transform(X_feature[['Grade']])

        X_feature_clean_per=std.fit_transform(X_feature[["Percentage"]])

        x_feature_train=pd.DataFrame(np.hstack([x_feature_grade_encode,X_feature_clean_per]))

        # Encode target as 1D array (no DataFrame wrapping)
        y_output_clean = lbl.fit_transform(Y_output["Result (Pass / Fail)"])

        x_feature_train=pd.DataFrame(np.hstack([x_feature_grade_encode,X_feature_clean_per]))

        loges_model=LogisticRegression()

        from sklearn.model_selection import train_test_split

        x_train,x_test,y_train,y_test=train_test_split(x_feature_train,y_output_clean,test_size=0.3,random_state=42)

        ##predict
        model_train=loges_model.fit(x_train,y_train)
        pred=model_train.predict(x_test)
        print(f"test : {x_test}  pred {pred}")


        score=model_train.score(x_test,y_test)
        print("score " ,score)


        ## model evaluation
        ##accurecy, recall , prescion, r2
        from sklearn.metrics import accuracy_score,precision_score,recall_score,r2_score
        acc=accuracy_score(y_test,pred)
        pr=precision_score(y_test,pred)
        rec_sc=recall_score(y_test,pred)
        r2_sc=r2_score(y_test,pred)

        print(f"acc {acc} , pr{pr} , rec_sc {rec_sc} , r2_sc {r2_sc}")


from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
def logisticRegression_direct() :
    X_feature=std_df[["Percentage","Grade"]]
    Y_output=std_df[["Result (Pass / Fail)"]]

    grade_hierarchy = ['U', 'D', 'C', 'B', 'A','A+']
    encoder = OrdinalEncoder(categories=[grade_hierarchy])
    x_feature_grade_encode = encoder.fit_transform(X_feature[['Grade']]) 

    X_feature['Grade']=x_feature_grade_encode
    x_train,x_test,y_train,y_test=train_test_split(X_feature,Y_output,test_size=0.2,random_state=42)
    logst_model=LogisticRegression()    
    logst_model.fit(x_train,y_train)
    pred=logst_model.predict(x_test)
    print(f"predection {pred}")
    score=logst_model.score(x_test,y_test)
    print(f"score {score}")

    knn_neigh_model=KNeighborsClassifier(n_neighbors=5)
    knn_neigh_model.fit(x_train,y_train)
    knn_pred=knn_neigh_model.predict(x_test)
    print(f"knn predection {knn_pred}")
    score=knn_neigh_model.score(x_test,y_test)
    print(f"knn score {score}")


logisticRegression_direct()

