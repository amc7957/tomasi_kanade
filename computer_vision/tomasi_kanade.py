from feature_detection import feature_detection
import numpy as np
from numpy import linalg

F = 4
frame_list = ['./input/castle.000.jpg','./input/castle.001.jpg','./input/castle.002.jpg','./input/castle.003.jpg']
#turn feature_detection.py into s function, run in loop to fill up U and V

U_array = []
V_array = []
for i in range(F):
    U,V = feature_detection(frame_list[i])

    P = len(U)
    a_f = [int((1/P)*sum(U))]*P
    b_f = [int((1/P)*sum(V))]*P


    U_new = [i - j for i, j in zip(a_f,U)]
    V_new = [k - l for k, l in zip(b_f,V)]

 
    U_array.append(U_new)
    V_array.append(V_new)

#Registered Measurement Matrix
W_new = (U_array[0],U_array[1],U_array[2],U_array[3],V_array[0],V_array[1],V_array[2],V_array[3])

u,s,vh = np.linalg.svd(W_new,full_matrices=True)
s = np.diag(s)
zeros = np.zeros((2*F,len(U)-2*F))
s = np.append(s,zeros)
s = np.reshape(s,(8,65))
#print(u)
#rank constraint
u_1 = u[:,0:3]
print(u_1)
vh_1 = vh[:,0:3]
s_1 = s[0:3,0:3]

M = np.dot(u_1,np.sqrt(s_1))
#print(M)
S = np.dot(vh_1,np.sqrt(s_1))

Q = []
R = []

for i in range(F):
    G = M[2*i-2,:]
    a = G[0]
    b = G[1]
    c = G[2]

    G = M[2*i-1,:]
    d = G[0]
    e = G[1]
    f = G[2]

print('done')    

    # Q = [Q
    #      a*a a*b+b*a a*c+c*a b*b c*b+b*c c*c,
    #      d*d d*e+e*d d*f+f*d e*e f*e+e*f f*f,
    #      a*d a*e+b*d a*f+c*d b*e b*f+e*c c*f]
        
    # R = [R, 1, 1, 0]
