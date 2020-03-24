# MODULE IMPORT ==============================================
import numpy as np

#=============================================================

# Let's simulate the system writing the hamiltonian in terms of spin operators

pauli_operators = np.array(( ((0,1),(1,0)),((0,-1j),(1j,0)),((1,0),(0,-1))  ))

# let's define the creation and distruction operators for a two site chain
# creation operator
c = (pauli_operators[0]+1j*pauli_operators[1])/2
# destruction operator
d = (pauli_operators[0]-1j*pauli_operators[1])/2



# operator declaration

c_2u = np.kron(np.identity(2),np.kron(np.identity(2),np.kron(np.identity(2),c)))
d_2u = np.kron(np.identity(2),np.kron(np.identity(2),np.kron(np.identity(2),d)))

c_2d = np.kron(np.identity(2),np.kron(np.identity(2),np.kron(c,pauli_operators[2])))
d_2d = np.kron(np.identity(2),np.kron(np.identity(2),np.kron(d,pauli_operators[2])))

c_1u = np.kron(np.identity(2),np.kron(c,np.kron(pauli_operators[2],pauli_operators[2])))
d_1u = np.kron(np.identity(2),np.kron(d,np.kron(pauli_operators[2],pauli_operators[2])))

c_1d = np.kron(c,np.kron(pauli_operators[2],np.kron(pauli_operators[2],pauli_operators[2])))
d_1d = np.kron(d,np.kron(pauli_operators[2],np.kron(pauli_operators[2],pauli_operators[2])))

dict={(1,"c","u"):c_1u,(1,"c","d"):c_1d,(1,"d","u"):d_1u,(1,"d","d"):d_1d,
      (2,"c","u"):c_2u,(2,"c","d"):c_2d,(2,"d","u"):d_2u,(2,"d","d"):d_2d}

# anticommutation function
def anti_comm(a,b):
    return (np.dot(a,b) + np.dot(b,a))

#for i in dict:
#    for j in dict:
#       print('{'+str(i[1])+'_'+str(i[0])+i[2]+' , '+ str(j[1])+'_'+str(j[0])+j[2]+ '}')
#       print(anti_comm(dict[i],dict[j]).trace())
#
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


# functions:

