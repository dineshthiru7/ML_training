import numpy as np

oneDArr=np.array([1,2,3,4,5,6,10])
twoDarr=np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9]
    ])

##indexing
print("one d array ",oneDArr[2])

##slicing
print("one d array ",oneDArr[0:4])
print("two d array : ", twoDarr[0:3])

print("two d array row and col : ", twoDarr[0:3,0:1]) ## 0:3 -> row and 0:1-> col
print("two d array row and col : ", twoDarr[0:3,0:2])

##boolean indexing
greaterTwo=oneDArr > 3
lessFive=twoDarr < 5
print("boolean indexing : ",greaterTwo)
print("boolean index two d : ", lessFive)
print("values print : ", twoDarr[lessFive])

## direct iterating
print("print less than 5 : ", oneDArr[oneDArr < 5])
