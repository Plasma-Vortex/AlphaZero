import os

NN_name = 'Conv v2'
files = os.listdir('NNs/'+NN_name)
ages = [int(f[len(NN_name)+2:-3]) for f in files]
maxAge = max(ages)
removed = 0
for i in ages:
    if i%100 != 0 and i < maxAge:
        os.remove('NNs/' + NN_name + '/' + NN_name + ', ' + str(i) + '.h5')
        removed += 1
print('%d files removed' % removed)