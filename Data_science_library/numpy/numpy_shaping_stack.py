import numpy as np

oneDArr=np.array([1,2,3,4,5,6,10,11,12])
twoDarr=np.array([
    [1,2,3],
    [4,5,6],
    [7,8,9],
    [10,11,12],
    ])

##shaping singled arry to multi d array
print("shapping array : ", oneDArr.reshape(3,3))
print("two d reshapping : ",twoDarr.reshape(3,4))

#ravel multi d to single array
print("ravel : ", twoDarr.ravel())
print("ravel : ", oneDArr.ravel())

ravelx=twoDarr.ravel()
ravelx[0]=100
ravelx[1]=200
print("after changes : ",ravelx)
print("orginal array : ",twoDarr)

#flattern as same as ravel
flattern=twoDarr.flatten()
print("flatteren : ",flattern)


arr1=np.array([1,2,3,4])
arr2=np.array([5,6,7,8])
#hstack
hstack=np.hstack([arr1,arr2])
print("hstqck : ",hstack)
#vstack
print("vstack : ", np.vstack([arr1,arr2,arr1]))

#transpose -> row move to col || col moven to row

trans=twoDarr.T
print("transpose : ",trans)
trans2=twoDarr.T.T
print("transpose : ", trans2)