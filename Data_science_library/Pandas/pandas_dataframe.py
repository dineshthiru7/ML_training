import pandas as pds
import numpy as np

## 2D aray for storing like table and excel data structure

data=[['ramesh','suresh'],[20,21]]
ds1=pds.DataFrame(data,index=['Name','age'])
print("basic ds :\n",ds1)

class_obj={
    'name':['nandhitha','thiru','dinesh','pickachu'],
    'age':[25,26,27,28],
    'is_pass':[True,False,True,False],
    'fees':[10,25.5,32,36.8]
}

ds2=pds.DataFrame(class_obj,index=[101,102,103,104])
print('ds2 : \n',ds2)

ds2['pincode']=[601,602,603,604]
ds2['fav_sub']=['maths','eng','prg','chem']
print(" after edit \n",ds2)

del ds2['fav_sub']
print("\n after delete \n",ds2)

ds3=ds2.drop('age',axis=1)
ds3=ds2.drop(101,axis=0)

print("\n ds3 \n", ds3)