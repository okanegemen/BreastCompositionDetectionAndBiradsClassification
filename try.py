a = [1,2,3,4,5]

b = [sum(a[:i]+a[i+1:]) if i!=len(a) else sum(a) for i in range(len(a))]

print(b)