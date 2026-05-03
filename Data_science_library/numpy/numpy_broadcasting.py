import numpy as nump

oneDArray=nump.array([1,2,3])
oneArr=[2]
twoDarray=nump.array([[1,2,3],[4,5,6]])

x=oneDArray+twoDarray
print("broad casting : ", x)

x=oneDArray-twoDarray
print("sub broad casting : ",x)

x=oneArr+oneDArray
print("array : ", x)

# issueArray=nump.array([2,3])
# x=issueArray+oneDArray
# print("issue broad casting : ",x)