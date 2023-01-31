a = {1:[3,2],2:[1,2],1:[9,1],2:[8,6]}
print(a[[1,2]])
for i in zip(*(a[[1,2]].values())):
    print(i)