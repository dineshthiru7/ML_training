import pandas as pds
import numpy as np

## it should be single diemensional array

arr=np.array(np.random.randint(10,20,3))
ser=pds.Series(arr)
print("core pds :\n", ser)
print("-----------")
ser=pds.Series(arr,index=[101,102,103])
print("series :\n",ser)

##print("zero post ", ser[0])
print("using custom index : ",ser[101])

##dictionaries
emp={
    "name" :"Nandhitha",
    "age":25,
    "salary":101.55,
    "pincode" : 601,
    "knows_ai":True
}

ser_dict=pds.Series(emp)
print("Employee Data \n",ser_dict)
print("")
print("Zero : ",ser_dict['name'])


## adding two series
ser1=pds.Series([1,2,3],index=[101,102,103])
ser2=pds.Series([4,5,6,7],index=[101,102,103,104])
print("series sum : ",ser1+ser2)
print("ser ",ser1.iloc((101)))
