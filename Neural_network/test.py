d= dict()
for i in range(1,5+1):
    d['string{}'.format(i)]=i
print(d)


for k in range(5):
    exec(f'cat_{k} = k*2')
print(cat_1)