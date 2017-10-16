from numpy import *
import sys

A = []
for line in open(sys.argv[1]):
    A.append(map(float, line.strip().split()))

B = []
for line in open(sys.argv[2]):
    B.append(map(float, line.strip().split()))

A = array(A)
B = array(B)
print dot(A, B)
