# MODULE IMPORT ==============================================
import math, numpy as np, scipy, pandas as pd, matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy.linalg import matrix_power
from mpl_toolkits.mplot3d import Axes3D

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer, IBMQ, BasicAer
from qiskit.compiler import transpile, assemble
from qiskit.tools.monitor import job_monitor
from qiskit.tools.jupyter import *
from qiskit.visualization import *

from benchmark_code import diagonalization as di

from qiskit.quantum_info import Kraus, SuperOp
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey
import json
#===========================================================


# CLASS: QUANTUM CALCULATION ===================================
# Read initial parameters and build Quantum Hubbard circuit
def hubb_buildQuantum(initstate, delta, T, V, nTrot):

    # Quantum circuit initialization
    q_fin = QuantumRegister(4)
    c_fin = ClassicalRegister(4)
    hubb_fin = QuantumCircuit(q_fin,c_fin)

    # Initial state definition
    if initstate == 1001:
        hubb_fin.x(3);
        hubb_fin.x(0);
    elif initstate == 1100:
        hubb_fin.x(3);
        hubb_fin.x(2);

    # Optimized circuit creation (no sigma_z in hopping term)

    # Trotter subcircuits ============
    q = QuantumRegister(4)
    hubb = QuantumCircuit(q)
    pi = np.pi;
    t_delta = 0.5;
    v_delta = 0.25;

    hubb.ry(-pi/2,0); # affetta da errori coerenti e incoerenti
    hubb.ry(-pi/2,1);
    hubb.ry(-pi/2,2);
    hubb.ry(-pi/2,3);
    hubb.cx(0,1);
    hubb.cx(2,3);
    hubb.rz(T*t_delta*2*delta,1);
    hubb.rz(T*t_delta*2*delta,3);
    hubb.cx(0,1);
    hubb.cx(2,3);
    hubb.ry(pi/2,0);
    hubb.ry(pi/2,1);
    hubb.ry(pi/2,2);
    hubb.ry(pi/2,3);
    hubb.rx(-pi/2,0);
    hubb.rx(-pi/2,1)
    hubb.rx(-pi/2,2);
    hubb.rx(-pi/2,3);
    hubb.cx(0,1);
    hubb.cx(2,3);
    hubb.rz(T*t_delta*2*delta,1);
    hubb.rz(T*t_delta*2*delta,3);
    hubb.cx(0,1);
    hubb.cx(2,3);
    hubb.rx(pi/2,0);
    hubb.rx(pi/2,1);
    hubb.rx(pi/2,2);
    hubb.rx(pi/2,3);
    hubb.cx(0,3);
    hubb.cx(1,2);
    hubb.rz(V*v_delta*2*delta,2);
    hubb.rz(V*v_delta*2*delta,3);
    hubb.cx(1,2);
    hubb.cx(0,3);
    hubb.rz(V*v_delta*2*delta,0);
    hubb.rz(V*v_delta*2*delta,1);
    hubb.rz(V*v_delta*2*delta,2);
    hubb.rz(V*v_delta*2*delta,3);
    # End Trotter subcircuits ============

    '''
    # Ry noisy (pi/2)
    hubb.u1(eps["X"],0)
    hubb.u3(-pi/2-eps["Z"],-pi/2,pi/2,0)
    hubb.u1(-pi/2,0)
    hubb.u3(pi/2+eps["Z"],-pi/2,pi/2,0)
    hubb.u1(-eps["X"],0)
    '''
    '''
    # Rx noisy (pi/2)
    hubb.u1(-eps["Y"],0)
    hubb.u3(-np.pi/2+eps["Z"],0.0,0.0,0)
    hubb.u1(pi/2,0)
    hubb.u3(np.pi/2-eps["Z"],0.0,0.0,0)
    hubb.u1(eps["Y"],0)
    '''

    # Append trotter subcircuits
    for i in range(nTrot): hubb_fin.append(hubb, [q_fin[0], q_fin[1], q_fin[2], q_fin[3]])

    # Measure final output
    hubb_fin.measure(0,0)
    hubb_fin.measure(1,1)
    hubb_fin.measure(2,2)
    hubb_fin.measure(3,3)

    return hubb_fin


# Read run parameters and execute circuit on simulator or on real HW
def hubb_executeQuantum(hubb_fin, delta, delta_range, BK, HW, opt_level, initial_layout, nShots):


    # Example error probabilities
    p_reset = 0.003
    p_meas = 0.1
    p_gate1 = 0.005

    # QuantumError objects
    error_reset = pauli_error([('X', p_reset), ('I', 1 - p_reset)])
    error_meas = pauli_error([('X',p_meas), ('I', 1 - p_meas)])
    error_gate1 = pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
    error_gate2 = error_gate1.tensor(error_gate1)

    # Add errors to noise model
    noise_bit_flip = NoiseModel()
    noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
    noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
    noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
    noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])

    # Local simulation
    if BK == 1:
        local_sim = BasicAer.get_backend('qasm_simulator')
        print('Run on local simulator '+str(local_sim)+' ...')
        # Transpile and run
        hubb_tran = transpile(hubb_fin, backend = local_sim, optimization_level = opt_level)
        job_sim = execute(hubb_tran, shots = nShots, backend = local_sim,  parameter_binds = [{delta: delta_val} for delta_val in delta_range])

    # Aer Noisy simulation
    if BK == 2:
        #local_sim = BasicAer.get_backend('qasm_simulator')
        aer_sim = QasmSimulator()
        print('Run on local simulator '+str(aer_sim)+' ...')
        # Transpile and run
        hubb_tran = transpile(hubb_fin, backend = aer_sim, optimization_level = opt_level)
        job_aer = execute(hubb_tran, shots = nShots, backend = aer_sim,  basis_gates=noise_bit_flip.basis_gates, noise_model=noise_bit_flip, parameter_binds = [{delta: delta_val} for delta_val in delta_range])

    # HW run
    elif BK == 3:
        cloud_hw = IBMQ.get_provider().get_backend(HW)
        print('Run on real hardware '+str(cloud_hw)+' ...')
        # Transpile and run
        if opt_level == 0:   hubb_tran = transpile(hubb_fin, backend = cloud_hw, initial_layout = initial_layout)
        elif opt_level != 0: hubb_tran = transpile(hubb_fin, backend = cloud_hw, optimization_level = opt_level)
        job_hw = execute(hubb_tran, shots = nShots, backend = cloud_hw, parameter_binds = [{delta: delta_val} for delta_val in delta_range])
        # Job monitoring
        job_hw_id = job_hw.job_id()
        job_monitor(job_hw)
        backend_properties = cloud_hw.properties()
        job_hw_date = job_hw.creation_date()

    # Collect results
    if BK == 1: counts = [job_sim.result().get_counts(i) for i in range(len(job_sim.result().results))]
    elif BK == 2: counts = [job_aer.result().get_counts(i) for i in range(len(job_aer.result().results))]
    elif BK == 3: counts = [job_hw.result().get_counts(i) for i in range(len(job_hw.result().results))]

    depth = hubb_tran.decompose().depth()
    gates = hubb_tran.decompose().count_ops()

    print('Run done.')

    results = dict()
    results[0] = counts
    results[1] = depth
    results[2] = gates
    if BK == 3:
        results[3] = job_hw_id
        results[4] = backend_properties
        results[5] = job_hw_date
    return results


# Count only feasible states and calculate correction factor
def hubb_calcMitigateObservables(results, nShots, corr):

    res_df = pd.DataFrame()
    for i in results[0]: res_df = res_df.append(i,ignore_index=True)
    res_df = res_df.fillna(0)

    # Shots on feasible and on unfeasible states
    feas_shots = res_df[['0110', '0101', '1001', '1010']].sum(axis=1)
    unfeas_shots = res_df[['0000', '0001', '0010', '0011', '0100', '0111', '1000', '1011', '1100', '1101', '1110', '1111']].sum(axis=1)

    # Consolidate shots on feasible and unfeasible states and calculate means
    counts_df = pd.concat([feas_shots.reset_index(), unfeas_shots.reset_index()], axis=1).drop(columns='index')
    counts_df.columns = ['feasible', 'unfeasible']
    counts_df['ratio'] = counts_df.unfeasible/counts_df.feasible
    counts_df.mean()

    # Correction factor
    if corr == 1: bound = counts_df.mean()['feasible']/counts_df.mean()['ratio'] # bound = feasible*(feasible/unfeasible)
    elif corr == 0: bound = nShots

    # Mapping
    # q3 q2 q1 q0
    # 1u 2u 2d 1d

    # Observables
    # (1-Sz)/2

    # Calculate and correct observables
    res_df["n1_up"] = (1-((res_df['0110']+res_df['0101'])-(res_df['1001']+res_df['1010']))/bound)/2
    res_df["n1_dn"] = (1-((res_df['0110']+res_df['1010'])-(res_df['1001']+res_df['0101']))/bound)/2
    res_df["n2_up"] = (1-((res_df['1010']+res_df['1001'])-(res_df['0101']+res_df['0110']))/bound)/2
    res_df["n2_dn"] = (1-((res_df['1001']+res_df['0101'])-(res_df['0110']+res_df['1010']))/bound)/2

    # Calculate particle number
    res_df["n_sum"] = (res_df["n1_up"]+res_df["n2_up"]+res_df["n2_dn"]+res_df["n1_dn"])

    return res_df


# Make classical and quantum plots
#def hubb_makePlots():


# CLASS: CLASSICAL BENCHMARK ====================================
# Calculate classical fermionic Hamiltonian temporal evolution
def hubb_calcClassical(initstate, time_range, T, V):

    # Declare initial state, for classical comparison it must be the same as quantum
    if initstate == 1001: in_st = di.initial_state(di.p,di.p,di.n,di.n)
    elif initstate == 1100: in_st = di.initial_state(di.p,di.n,di.n,di.p)

    # Analytical (fermionic) ===============
    result = []
    for i in time_range:
        fin_state = np.dot(expm(-i*1j*di.H(T,V)),in_st)
        result.append(np.real(fin_state.conjugate().transpose().dot(di.n_1u.dot(fin_state))))

    results = dict()
    results[0] = result
    return results


# Calculate classical spin Hamiltonian temporal evolution
def hubb_calcClassicalSpin(initstate, time_range, T, V):

    # Declare initial state, for classical comparison it must be the same as quantum
    if initstate == 1001: in_st = di.initial_state(di.p,di.p,di.n,di.n)
    elif initstate == 1100: in_st = di.initial_state(di.p,di.n,di.n,di.p)

    # Analytical spin no trotter =============
    result_spin = []
    H_k = H_v = 0
    for i in range(4): H_k += di.K[i]
    for i in range(6): H_v += di.I[i]

    H_t = -T*H_k/2 + V*H_v/4 # understand why it works with these coefficients
    for i in time_range:
        fin_state1 = np.dot(expm(-i*1j*H_t),in_st)
        result_spin.append(np.real(fin_state1.conjugate().transpose().dot(di.n_1u.dot(fin_state1))))

    results = dict()
    results[0] = result_spin
    return results


# Calculate classical spin trotterized Hamiltonian temporal evolution
def hubb_calcClassicalSpinTrot(initstate, time_range, T, V, nTrot):

    # Declare initial state, for classical comparison it must be the same as quantum
    if initstate == 1001: in_st = di.initial_state(di.p,di.p,di.n,di.n)
    elif initstate == 1100: in_st = di.initial_state(di.p,di.n,di.n,di.p)

    # Analytical spin trotter ================
    result_tr = []
    H_fin = []
    for t in time_range:
        H_k = np.dot(expm(-1j*T*t*di.K[0]/nTrot/2),\
                    np.dot(expm(-1j*T*t*di.K[1]/nTrot/2),\
                    np.dot(expm(-1j*T*t*di.K[2]/nTrot/2),\
                                 expm(-1j*T*t*di.K[3]/nTrot/2))))
        H_v = np.dot(expm(-1j*V*t*di.I[0]/nTrot/4),\
                    np.dot(expm(-1j*V*t*di.I[1]/nTrot/4),\
                    np.dot(expm(-1j*V*t*di.I[2]/nTrot/4),\
                    np.dot(expm(-1j*V*t*di.I[3]/nTrot/4),\
                    np.dot(expm(-1j*V*t*di.I[4]/nTrot/4),\
                                 expm(-1j*V*t*di.I[5]/nTrot/4))))))

        H_fin = np.dot(H_k, H_v)
        H_fin = matrix_power(H_fin, nTrot)
        fin_state_tr = np.dot(H_fin, in_st)
        result_tr.append(np.real(fin_state_tr.conjugate().transpose().dot(di.n_1u.dot(fin_state_tr))))

    results = dict()
    results[0] = result_tr
    return results



# CLASS: SAVA DATA TO DB =========================================
# Save data to DB
def hubb_saveDB(Document, serviceUsername, servicePassword, serviceURL, databaseName):

    # Connect
    client = Cloudant(serviceUsername, servicePassword, url=serviceURL)
    client.connect()

    myDatabaseDemo = client[databaseName]
    if myDatabaseDemo.exists(): print("DB ready.")

    strDocument = json.dumps(Document)
    jsonDocument = json.loads(strDocument)

    # Save document to Cloudant
    newDocument = myDatabaseDemo.create_document(jsonDocument)

    # Check
    if newDocument.exists(): print("Document created.")


# Retrieve data from DB
def hubb_retrieveDB(DB_id, serviceUsername, servicePassword, serviceURL, databaseName):

    # Connect
    client = Cloudant(serviceUsername, servicePassword, url=serviceURL)
    client.connect()

    myDatabaseDemo = client[databaseName]
    if myDatabaseDemo.exists(): print("DB ready.")

    # Retrieve from DB is required
    if DB_id != '':

        run_DB = myDatabaseDemo[DB_id]
        run_json = json.dumps(run_DB)
        run = json.loads(run_json)

        # Extract data
        data = dict()
        results = dict()

        data[0] = run['T']
        data[1] = run['V']
        data[2] = run['HW']
        data[3] = run['nTrot']
        data[4] = run['nShots']
        data[5] = run['step_c']
        data[6] = run['step_q']
        data[7] = run['opt_level']
        data[8] = run['initial_layout']
        data[9] = run['initstate']
        data[10] = run['res_exact']
        data[11] = run['res_spin']
        data[12] = run['res_spin_trot']

        results[0] = run['counts']
        results[1] = run['depth']
        results[2] = run['gates']
        results[3] = run['job_id']
        results[4] = run['backend_properties']
        results[5] = run['job_date']

        print('Data retrieved from DB.')

        return [results,data]
