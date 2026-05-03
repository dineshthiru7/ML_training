import numpy as np

oneDArr=np.array([1,2,3,4,5,6,10])
twoDarr=np.array([[1,2,3],[4,5,6],[7,8,9]])

print("sum array : ", np.sum(oneDArr))
print("two d sum : ", np.sum(twoDarr))

print("mean : ", np.mean(oneDArr)) ## average 
print("median : ",np.median(oneDArr)) ## center point

print("min : ", oneDArr.min())
print("max : ",oneDArr.max())

print("two d min ", twoDarr.min())
print("three d max ", twoDarr.max())

## variants 
print("variants : ",oneDArr.var())
print("standard devation ",oneDArr.std())