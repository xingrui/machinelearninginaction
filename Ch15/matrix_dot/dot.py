from numpy import *

A = []
for line in open('left.txt.org'):
    A.append(map(float, line.strip().split()))

B = []
for line in open('right.txt.org'):
    B.append(map(float, line.strip().split()))

A = array(A)
B = array(B)
print A
print B
print dot(A, B)
