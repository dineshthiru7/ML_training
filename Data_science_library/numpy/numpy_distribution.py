import numpy as np
import matplotlib.pyplot as plt


## normal distribution
## uniform distribution
## binominal distribution

##rand -> this is give vlues between 0 to 1
print("rant 1 ", np.random.rand())
print("rant 2 ", np.random.rand())
print("rant 3 ", np.random.rand())

##randint
randint=np.random.randint(100,300,size=100)

## normal distribution : like this is hill or bell curve

np.random.seed(42)
arr=np.array([randint])

average_value=np.mean(arr)
stndrd_devation=np.std(arr)
total=np.size(arr)
print(f"average {average_value}  , std devation {stndrd_devation} , total {total}")

normalDist=np.random.normal(loc=average_value,scale=stndrd_devation,size=total)
##print("normal devtion : ",normalDist)

##matplot lib --> Histogram create a histogram, the array is sent into the function as an argument.

x=False
if x :
    plt.hist(normalDist)
    plt.xlabel("date")
    plt.ylabel("sales")
#plt.show()

## uniform distribution ->  equal chances of occuring. like die rotation possible to give (1 to 6 )
unifor_distribution=np.random.uniform(low=1,high=6,size=6000)
if x :
    plt.hist(unifor_distribution)
    plt.xlabel("dies number")
    plt.ylabel("dies rotate count")
##plt.show()

##bionominal distribution  e.g. toss of a coin, it will either be head or tails.
# n - number of trails
#p = probability of occurrence of each trial (e.g. for toss of a coin 0.5 each).
#size = The shape of the returned array.
bionomial_distribution=np.random.binomial(n=100,p=0.5,size=100)
plt.hist(bionomial_distribution)
plt.xlabel("trail")
plt.ylabel("total count")
plt.show()

