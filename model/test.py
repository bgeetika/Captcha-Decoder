import numpy as np


f1 = np.load("/home/sujeetb/geetika/dataset/clipping_more_lstm_2015_11_23_18_04_57.npy.npz")
f2 = np.load("/home/sujeetb/geetika/dataset/clipping_bidirec_more_lstm_2015_11_23_18_44_50.npy.npz")
p1 = [f1['arr_%d' % i] for i in range(len(f1.files))]
p2 = [f2['arr_%d' % i] for i in range(len(f2.files))]



for x in range(0, min(len(p1),len(p2))):
    print p1[x].shape, p2[x].shape
    if p1[x].shape == p2[x].shape:
        p2[x] = p1[x]

print min(len(p1),len(p2))
print x
print "deone withsimilar"

for y in range(x, len(p2)):
    print p2[y].shape

for x in range(8,23):
    p2[x+17] = p1[x] 



np.savez("geet.npy.npz", *p2)

f3 = np.load("geet.npy.npz")
p3 = [f3['arr_%d' % i] for i in range(len(f3.files))]


for x in range(0, len(p2)):
    if p2[x].shape != p3[x].shape:
         print "not deone properly"
