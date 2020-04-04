from qiskit import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from qiskit import IBMQ
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import Aer, execute
from qiskit.aqua.components.optimizers import COBYLA, SLSQP, ADAM
from qiskit import BasicAer
from qiskit.quantum_info import Pauli
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import ExactEigensolver, VQE
from qiskit.aqua.components.variational_forms import RY, RYRZ
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.operators import WeightedPauliOperator

backend = Aer.get_backend("qasm_simulator")
shots =8192
J=1
#b = np.linspace(1/4,4,1)
b=[1/4,1/2,1,2,2.5,3,3.5,4]
ret=[]
par=[]
Magx =[]
optimizer = COBYLA(maxiter=2000,rhobeg =2)
#optimizer = SLSQP(maxiter=1000000, tol=0.0000000001)
#optimizer = ADAM(maxiter=200, tol=0.000001, lr=0.8, eps = 1*1e-5)
def Magnetizzazione(counts):
    if not "00" in counts: counts["00"] = 0
    if not "11" in counts: counts["11"] = 0
    mean = (counts["00"] - counts["11"])/shots
    return mean

for k in range(len(b)): # Crea gli operatori da Pauli da misurare pesati
  op=[]
  op.append((J,Pauli(label = 'ZZ')))
  op.append((J,Pauli(label = 'XX')))
  op.append((J,Pauli(label = 'YY')))
  op.append((b[k],Pauli(label = 'ZI')))
  op.append((b[k],Pauli(label = 'IZ')))


  Operator = WeightedPauliOperator(op, basis=None, z2_symmetries=[0,1], atol=1e-12, name=None)
  var_form = RY(num_qubits=2, depth=2, entanglement='full', entanglement_gate='cx', skip_unentangled_qubits=False, skip_final_ry=False)
  #var_form = RYRZ(num_qubits=2, depth=2, entanglement='full')
  vqe = VQE(Operator, var_form, optimizer)

  p = vqe.run(backend)['opt_params']
  vqe.get_optimal_circuit()
  par.append(p)
  ret.append(vqe.run(backend)['energy'])# salvo le energie minime

  qc = QuantumCircuit(2,2)
  qc.append(var_form.construct_circuit(p),[0,1]) #Stato a minima Energia
  qc.h(0)
  qc.h(1)
  qc.measure([0,1],[0,1])
  job = execute(qc, backend=backend,shots= shots)
  result = job.result()
  counts = result.get_counts(qc)
  Magx.append(Magnetizzazione(counts))


print(ret)
print(Magx)

#ig, ax = plt.subplots()
#plt.xticks()
#ax.plot(b,ret,marker='o')
#ax.set(xlabel='b', ylabel='E',
#q       title='Grafico campo-energia (RY)')
#ax.grid()
#plt.show()
