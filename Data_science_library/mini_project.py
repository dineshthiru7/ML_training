import numpy as np
import pandas as pd


np.random.seed(42)
data1=np.random.randint(60,101,size=(40,5))
data2=np.random.randint(1,44,size=(30,5))
data3=np.random.randint(44,60,size=(30,5))
data=np.vstack((data1,data2,data3))
print("data : ",data)

##Apply grace marks to all students pass mark is 45 and grace 
grace_boolean=((data > 42 ) & (data < 45))
##print(grace_boolean)
data= np.where((data > 42) & (data < 45), data + (45-data) , data)
##print("data : ",data)

################Ensure marks do not exceed 100s
exceed_flag=data>100
exceed_data=data[exceed_flag]
if exceed_data.size >0 :
    print("data mis match , the mark should be less than 100 ")

###################Calculate:
#Mean marks subject-wise
mean_mark_subjectwise=np.mean(data,axis=0)
#Maximum marks
max_mark=np.max(data,axis=0)
#Minimum marks
min_mark=data.min(axis=0)
#Standard deviation
std_dev=data.std(axis=0)
#varients
varients=np.var(data,axis=0)

##Part 2 — Pandas Requirements
subject=["Math","Science","English","Computer","Social"]
dataFrame1=pd.DataFrame(data=data,index=[np.random.randint(1000,1100,100)],columns=subject)
##print(dataFrame1)
###O/P
######       Math  Science  English  Computer  Social
######1075    52       93       15        72      61


#Create new columns:
        
        
       
        
#Total Marks
dataFrame1["Total Marks"]=dataFrame1.sum(axis=1)

#Average Marks
dataFrame1["Average"]=dataFrame1[subject].mean(axis=1)

#Percentage
dataFrame1["Percentage"]=(dataFrame1["Total Marks"]/500)*100

#Result (Pass / Fail)
pass_flag=dataFrame1[subject] >= 45 
pass_flag_1d = pass_flag.all(axis=1)
dataFrame1["Result (Pass / Fail)"] = np.where(pass_flag_1d, "Pass", "Fail")

#Grade
##Pass -> 45 - 60 : C , 61 - 90 : A , 90+ :A+ 
##Fail -> U
# 1. List your conditions
conditions = [
    (dataFrame1["Result (Pass / Fail)"] == "Fail"),                               # Rule for U
    (dataFrame1["Percentage"] > 45) & (dataFrame1["Percentage"] <= 60),            # Rule for C
    (dataFrame1["Percentage"] >= 61) & (dataFrame1["Percentage"] <= 90),           # Rule for A
    (dataFrame1["Percentage"] >= 90)                                                # Rule for A+
]

# 2. List the corresponding grades
choices = ["U", "C", "A", "A+"]

# 3. Apply it all at once
dataFrame1["Grade"] = np.select(conditions, choices, default="Incomplete")


################Data Operations

##Sort students by highest total
sort_based_total=dataFrame1.sort_values(by="Total Marks",ascending=False)

##Filter students scoring above 80 average
above_80_flag=dataFrame1["Percentage"] >= 80
score_above_80=dataFrame1[above_80_flag]

##Find top 10 students
top_10_stud=sort_based_total.head(10)

##Find failed students (if any)
failed_students=sort_based_total[sort_based_total["Grade"] =="U"]



##############Part 3 — CSV File Requirements

# export_csv=pd.DataFrame.to_csv(sort_based_total,"Data_science_library\output\std.csv")
# export_excel=pd.DataFrame.to_excel(sort_based_total,"Data_science_library\output\std_ex.xlsx")

############### Part 4 — Data Analysis Requirements


#Subject Analysis
        #Highest scoring subject
highest_score_subject=dataFrame1[subject].max()
        #Lowest scoring subject
low_score_subject=dataFrame1[subject].min()
        #Average marks per subject
avg_mark_subject=dataFrame1[subject].mean()

grade_flag=dataFrame1["Grade"]=="A"
agrade_size=len(dataFrame1[grade_flag])



# 🔹 Part 5 — Matplotlib Dashboard Requirements

import matplotlib.pyplot as plt

### Chart 1 — Bar Chart
top_10 = dataFrame1.nlargest(100, "Total Marks")
x_labels = [ "-".join(map(str, idx)) if isinstance(idx, tuple) else str(idx) for idx in top_10.index]
y_score = top_10["Math"].values
# plt.figure(figsize=(10, 6))
# plt.bar(x_labels, y_score, color='skyblue')
# plt.show()

### Chart 2 — Histogram

# plt.hist(y_score)
# plt.show()


### Chart 3 — Scatter Plot
plt.scatter(x_labels,y_score)
plt.show()


### Chart 4 — Pie Chart

grade_pass=dataFrame1["Grade"].value_counts()
# plt.pie(grade_pass)
# plt.show()