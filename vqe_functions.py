from qiskit import *
import numpy as np
from numpy import linalg
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import itertools

# Qiskit Terra
from qiskit import execute, Aer, IBMQ, BasicAer
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.monitor import job_monitor
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.quantum_info import Pauli

# Qiskit Aqua
from qiskit.aqua.components.optimizers import COBYLA, SLSQP, ADAM, SPSA
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import ExactEigensolver, VQE
from qiskit.aqua.components.variational_forms import RY, RYRZ
from qiskit.aqua.operators import WeightedPauliOperator

# Qiskit Aer
from qiskit.providers.aer import QasmSimulator

# Qiskit extensions
from qiskit.extensions import RXGate, CnotGate, XGate, IGate
​

​
def make_opt(opt_str):
    if opt_str == "spsa":
        optimizer = SPSA(max_trials=100, save_steps=1, c0=4.0, skip_calibration=True)
    elif opt_str == "cobyla":
        optimizer = COBYLA(maxiter=1000, disp=False, rhobeg=1.0, tol=None)
    elif opt_str == "adam":
        optimizer = ADAM(maxiter=10000, tol=1e-6, lr=1e-3, beta_1=0.9, beta_2=0.99, noise_factor=1e-8, eps=1e-10)
    else:
        print('error in building OPTIMIZER: {} IT DOES NOT EXIST'.format(opt_str))
        sys.exit(1)
    return optimizer



def make_varfor(var_str, feature_dim, vdepth):
    if var_str == "ryrz":
        var_form = RYRZ(num_qubits=feature_dim, depth=vdepth, entanglement='full', entanglement_gate='cz')
    if var_str == "ry":
        var_form = RY(num_qubits=feature_dim, depth=vdepth, entanglement='full', entanglement_gate='cz')
    else:
        print('error in building VARIATIONAL FORM {}'.format(var_str))
        sys.exit(1)
    return var_form



def create_operator(michele, ratio):
    # Crea gli operatori da Pauli da misurare pesati
    if michele == 0:
        op = []
        op.append((1/2,Pauli(label = 'XXII')))
        op.append((1/2,Pauli(label = 'YYII')))
        op.append((1/2,Pauli(label = 'IIXX')))
        op.append((1/2,Pauli(label = 'IIYY')))
        op.append((ratio/4,Pauli(label = 'ZIIZ')))
        op.append((ratio/4,Pauli(label = 'IZZI')))
        op.append((ratio/4,Pauli(label = 'ZIII')))
        op.append((ratio/4,Pauli(label = 'IZII')))
        op.append((ratio/4,Pauli(label = 'IIZI')))
        op.append((ratio/4,Pauli(label = 'IIIZ')))
    elif michele == 1:
        op = []
        op.append((1/2,Pauli(label = 'IYZY')))
        op.append((1/2,Pauli(label = 'IXZX')))
        op.append((1/2,Pauli(label = 'YZYI')))
        op.append((1/2,Pauli(label = 'XZXI')))
        op.append((-ratio/4,Pauli(label = 'IIIZ')))
        op.append((-ratio/4,Pauli(label = 'IZII')))
        op.append((ratio/4,Pauli(label = 'IZIZ')))
        op.append((-ratio/4,Pauli(label = 'IIZI')))
        op.append((-ratio/4,Pauli(label = 'ZIII')))
        op.append((ratio/4,Pauli(label = 'ZIZI')))

    operator = WeightedPauliOperator(op, basis=None, z2_symmetries=[0,1], atol=1e-12, name=None)
    return operator
