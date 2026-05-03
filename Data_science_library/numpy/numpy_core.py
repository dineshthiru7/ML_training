import numpy as np

print("Num Py Version : ",np.__version__)

oneDArray=np.array([1,2,3,4,5,6])
print("One diemnsional array : ",oneDArray)

twoDarr=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print("Two D array",twoDarr)


##shape
print("1d arr ",np.shape(oneDArray))
print("3d array ",np.shape(twoDarr))

##doem
print("1d array",np.ndim(oneDArray))
print("2d array : ", np.ndim(twoDarr))

##dtype
print("1d array Dtype",oneDArray.dtype)
print("2d array type ", twoDarr.dtype)

oneDArrayType=np.array([1,2,3,4,5],dtype=int)
print("data type ",oneDArrayType.dtype)

##The data flow will happends int -> float -> st

###ones , zeors , consttans

print("Zeros " , np.zeros((2,3)))
print("ones ", np.ones((4,4)))
print("constants ", np.full((3,3),10))
