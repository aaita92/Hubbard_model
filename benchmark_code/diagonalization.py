# MODULE IMPORT ==============================================
import numpy as np
import scipy
from scipy.linalg import expm

#=============================================================
# Let's define pauli operators and identity

pauli_operators = np.array(( ((0,1),(1,0)),((0,-1j),(1j,0)),((1,0),(0,-1)),((1,0),(0,1))  ))

Sx = pauli_operators[0]
Sy = pauli_operators[1]
Sz = pauli_operators[2]
Id = pauli_operators[3]



#SPIN EXACT SOLUTION ======================================
# Let's simulate the system writing the hamiltonian in terms of spin operators

# let's define the creation and distruction operators for a two site chain
# creation operator
c = (Sx+1j*Sy)/2
# destruction operator
d = (Sx-1j*Sy)/2

# operator declaration

c_2u = np.kron(Id,np.kron(Id,np.kron(Id,c)))
d_2u = np.kron(Id,np.kron(Id,np.kron(Id,d)))

c_2d = np.kron(Id,np.kron(Id,np.kron(c,Sz)))
d_2d = np.kron(Id,np.kron(Id,np.kron(d,Sz)))

c_1u = np.kron(Id,np.kron(c,np.kron(Sz,Sz)))
d_1u = np.kron(Id,np.kron(d,np.kron(Sz,Sz)))

c_1d = np.kron(c,np.kron(Sz,np.kron(Sz,Sz)))
d_1d = np.kron(d,np.kron(Sz,np.kron(Sz,Sz)))

dict={(1,"c","u"):c_1u,(1,"c","d"):c_1d,(1,"d","u"):d_1u,(1,"d","d"):d_1d,
      (2,"c","u"):c_2u,(2,"c","d"):c_2d,(2,"d","u"):d_2u,(2,"d","d"):d_2d}

# anticommutation function
def anti_comm(a,b):
    return (np.dot(a,b) + np.dot(b,a))

# single site number operator
n_1u = np.dot(c_1u,d_1u)
n_1d = np.dot(c_1d,d_1d)
n_2u = np.dot(c_2u,d_2u)
n_2d = np.dot(c_2d,d_2d)

# write hamiltonian
H = lambda t,v: -t*(np.dot(c_1u,d_2u) + np.dot(c_1d,d_2d) +  np.dot(c_2u,d_1u) + np.dot(c_2d,d_1d))\
                    + v*(np.dot(n_1d,n_1u) + np.dot(n_2d,n_2u))

# create a state
p = np.array([ 1 , 0])
n = np.array([0 , 1])

# define initial state
initial_state = lambda u,d,t,q: np.kron(u,np.kron(d,np.kron(t,q)))



#SPIN TROTTERIZED SOLUTION ======================================
# Let's simulate the system writing the hamiltonian in terms of spin operators

#MAPPING 1 WITH 3-BODIES COMPONENT

# K = [
#     ( np.kron( np.kron(Sx,Sz),np.kron(Sx,Id) ) ),
#     ( np.kron( np.kron(Sy,Sz),np.kron(Sy,Id) ) ),
#     ( np.kron( np.kron(Id,Sx),np.kron(Sz,Sx) ) ),
#     ( np.kron( np.kron(Id,Sy),np.kron(Sz,Sy) ) )
# ]
# I = [
#     ( np.kron( np.kron(Sz,Sz),np.kron(Id,Id) ) ),
#     ( np.kron( np.kron(Id,Id),np.kron(Sz,Sz) ) ),
#     ( np.kron( np.kron(Sz,Id),np.kron(Id,Id) ) ),
#     ( np.kron( np.kron(Id,Sz),np.kron(Id,Id) ) ),
#     ( np.kron( np.kron(Id,Id),np.kron(Sz,Id) ) ),
#     ( np.kron( np.kron(Id,Id),np.kron(Id,Sz) ) )
# ]

#MAPPING 1 NO 3-BODIES COMPONENT

K = [
    ( np.kron( np.kron(Sx,Id),np.kron(Sx,Id) ) ),
    ( np.kron( np.kron(Sy,Id),np.kron(Sy,Id) ) ),
    ( np.kron( np.kron(Id,Sx),np.kron(Id,Sx) ) ),
    ( np.kron( np.kron(Id,Sy),np.kron(Id,Sy) ) )
]
I = [
    ( np.kron( np.kron(Sz,Sz),np.kron(Id,Id) ) ),
    ( np.kron( np.kron(Id,Id),np.kron(Sz,Sz) ) ),
    ( np.kron( np.kron(Sz,Id),np.kron(Id,Id) ) ),
    ( np.kron( np.kron(Id,Sz),np.kron(Id,Id) ) ),
    ( np.kron( np.kron(Id,Id),np.kron(Sz,Id) ) ),
    ( np.kron( np.kron(Id,Id),np.kron(Id,Sz) ) )
]

#MAPPING 2

# K = [
#     ( np.kron( np.kron(Sx,Sx),np.kron(Id,Id) ) ),
#     ( np.kron( np.kron(Sy,Sy),np.kron(Id,Id) ) ),
#     ( np.kron( np.kron(Id,Id),np.kron(Sx,Sx) ) ),
#     ( np.kron( np.kron(Id,Id),np.kron(Sy,Sy) ) )
# ]
# I = [
#     ( np.kron( np.kron(Sz,Id),np.kron(Id,Sz) ) ),
#     ( np.kron( np.kron(Id,Sz),np.kron(Sz,Id) ) ),
#     ( np.kron( np.kron(Sz,Id),np.kron(Id,Id) ) ),
#     ( np.kron( np.kron(Id,Sz),np.kron(Id,Id) ) ),
#     ( np.kron( np.kron(Id,Id),np.kron(Sz,Id) ) ),
#     ( np.kron( np.kron(Id,Id),np.kron(Id,Sz) ) )
# ]
