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
def hubb_executeQuantum(hubb_fin, delta, delta_range, BK, HW, opt_level, initial_layout, nShots, noise_model, coupling_map):

    # Local simulation
    if BK == 1:
        local_sim = BasicAer.get_backend('qasm_simulator')
        print('Run on local simulator '+str(local_sim)+' ...')
        # Run
        job_sim = execute(hubb_fin, shots = nShots, backend = local_sim,  parameter_binds = [{delta: delta_val} for delta_val in delta_range])

    # Aer Noisy simulation
    if BK == 2:
        aer_sim = QasmSimulator()
        print('Run on Aer simulator '+str(aer_sim)+' ...')
        # Run
        job_aer = execute(hubb_fin, shots = nShots, backend = aer_sim, coupling_map = coupling_map, basis_gates = noise_model.basis_gates, noise_model = noise_model, parameter_binds = [{delta: delta_val} for delta_val in delta_range])

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
    elif BK == 3:
        counts = [job_hw.result().get_counts(i) for i in range(len(job_hw.result().results))]
        depth = hubb_tran.decompose().depth()
        gates = hubb_tran.decompose().count_ops()

    print('Run done.')

    results = dict()
    results[0] = counts
    if BK == 3:
        results[1] = depth
        results[2] = gates
        results[3] = job_hw_id
        results[4] = backend_properties
        results[5] = job_hw_date
    return results


# Implement error mitigation
# corr = 0: no correction
# corr = 1: only consider feasible_shots
# corr = 2: consider feasible_shots normalized
def hubb_mitigateErrors(line, obs, corr):

    feas_shots = line[['0110', '0101', '1001', '1010']].sum()
    unfeas_shots = line[['0000', '0001', '0010', '0011', '0100', '0111', '1000', '1011', '1100', '1101', '1110', '1111']].sum()

    myoutput = 0
    bound = 0
    # inserire anche simulazioni non corrette da fattore di scala

    if corr == 0:
        bound = line.sum() # all shots
        if obs == 'n1_up':
            #fact = 1#/0.6
            #myoutput = (1-(fact*(line['0000']-line['1000'] + line['0001']-line['1001'] + line['0010']-line['1010'] + line['0100']-line['1100'] + line['0110']-line['1110'] + line['0011']-line['1011'] + line['0111']-line['1111'] + line['0101']-line['1101']))/bound)/2
            myoutput = (1-((line['0110']+line['0101'])-(line['1001']+line['1010']))/bound)/2

    elif corr == 1:
        bound = feas_shots # only feasible states shots
        fact = 1
        if obs == 'n1_up': myoutput = (1-(fact*(line['0110']+line['0101'])-(line['1001']+line['1010']))/bound)/2
        elif obs == 'n1_dn': myoutput = (1-((line['0110']+line['1010'])-(line['1001']+line['0101']))/bound)/2
        elif obs == 'n2_up': myoutput = (1-((line['1010']+line['1001'])-(line['0101']+line['0110']))/bound)/2
        elif obs == 'n2_dn': myoutput = (1-((line['1001']+line['0101'])-(line['0110']+line['1010']))/bound)/2

    elif corr == 2:
        bound = (feas_shots*feas_shots)/unfeas_shots # abbiamo bisogno di riscalare considerando i count che sono confluiti negli stati unfeasible per via dell'errore
        if obs == 'n1_up': myoutput = (1-((line['0110']+line['0101'])-(line['1001']+line['1010']))/bound)/2
        elif obs == 'n1_dn': myoutput = (1-((line['0110']+line['1010'])-(line['1001']+line['0101']))/bound)/2
        elif obs == 'n2_up': myoutput = (1-((line['1010']+line['1001'])-(line['0101']+line['0110']))/bound)/2
        elif obs == 'n2_dn': myoutput = (1-((line['1001']+line['0101'])-(line['0110']+line['1010']))/bound)/2

    return myoutput


# Implement limit correction (no > 1 and no < 0)
# Line is a single result_df line
# Obs can be 'up' or 'dn'
def hubb_correctLimit(line, obs):

    # Spin up species
    if obs == 'up':

        if line.n1_up > 1:  # if n1_up is bigger than 1
            diff = 1 - line.n1_up
            return pd.Series([1,line.n2_up-diff])

        elif line.n2_up > 1: # if n2_up is bigger than 1
            diff = 1 - line.n2_up
            return pd.Series([line.n1_up-diff,1])

        else: return pd.Series([line.n1_up,line.n2_up])

    # Spin down species
    if obs == 'dn':

        if line.n1_dn > 1: # if n1_dn is bigger than 1
            diff = 1 - line.n1_dn
            return pd.Series([1,line.n2_dn-diff])

        elif line.n2_dn > 1: # if n2_dn is bigger than 1
            diff = 1 - line.n2_dn
            return pd.Series([line.n1_dn-diff,1])

        else: return pd.Series([line.n1_dn,line.n2_dn])


# Count only feasible states and calculate correction factor
def hubb_calcMitigateObservables(results, nShots, BK, corr, limit_corr):

    # Load calculation results on a DataFrame
    res_df = pd.DataFrame()
    for i in results[0]: res_df = res_df.append(i,ignore_index=True)
    res_df = res_df.fillna(0)

    # Error mitigation only if real HW run (BK = 3)
    # Calculate observables with correction
    res_df['n1_up'] = res_df.apply(lambda x:hubb_mitigateErrors(x,'n1_up',corr),axis=1)
    res_df['n1_dn'] = res_df.apply(lambda x:hubb_mitigateErrors(x,'n1_dn',corr),axis=1)
    res_df['n2_up'] = res_df.apply(lambda x:hubb_mitigateErrors(x,'n2_up',corr),axis=1)
    res_df['n2_dn'] = res_df.apply(lambda x:hubb_mitigateErrors(x,'n2_dn',corr),axis=1)

    if limit_corr == 1:
        # Limit correction (no > 1 and no < 0)
        res_df[['n1_up', 'n2_up']] = res_df.apply(lambda x:hubb_correctLimit(x,'up'),axis=1)
        res_df[['n1_dn', 'n2_dn']] = res_df.apply(lambda x:hubb_correctLimit(x,'dn'),axis=1)

    # Calculate particle number with limit correction
    res_df["n_sum"] = (res_df["n1_up"]+res_df["n2_up"]+res_df["n2_dn"]+res_df["n1_dn"])

    return res_df


# Create noise model and coupling map for noisy Aer simulations
def hubb_defineNoise(HW, times, p0g1, p1g0, dpol_u1, dpol_u2, dpol_u3, dpol_cx, err_flags):

    T1s = times[0]
    T2s = times[1]
    time_u1 = times[2]
    time_u2 = times[3]
    time_u3 = times[4]
    time_cx = times[5]
    time_reset = times[6]
    time_measure = times[7]

    # THERMAL RELAXATION ERROR
    if err_flags[0] == 1:
        errors_reset = [thermal_relaxation_error(t1, t2, time_reset) for t1, t2 in zip(T1s, T2s)]
        errors_measure = [thermal_relaxation_error(t1, t2, time_measure) for t1, t2 in zip(T1s, T2s)]
        errors_u1 = [thermal_relaxation_error(t1, t2, time_u1) for t1, t2 in zip(T1s, T2s)]
        errors_u2 = [thermal_relaxation_error(t1, t2, time_u2) for t1, t2 in zip(T1s, T2s)]
        errors_u3 = [thermal_relaxation_error(t1, t2, time_u3) for t1, t2 in zip(T1s, T2s)]
        errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(thermal_relaxation_error(t1b, t2b, time_cx))
                      for t1a, t2a in zip(T1s, T2s)]
                      for t1b, t2b in zip(T1s, T2s)]

    # READOUT ERROR
    if err_flags[1] == 1:
        readout = []
        for r in range(5):
            readout.append(ReadoutError([[1 - p1g0[r], p1g0[r]], [p0g1[r], 1 - p0g1[r]]]))

    # DEPOLARIZING ERROR
    if err_flags[2] == 1:
        for j in range(5):
            dpol_u1[j] = depolarizing_error(dpol_u1[j], 1)
            dpol_u2[j] = depolarizing_error(dpol_u2[j], 1)
            dpol_u3[j] = depolarizing_error(dpol_u3[j], 1)
            for k in range(5):
                try:
                    dpol_cx[j,k] = depolarizing_error(dpol_cx[j,k]['gate_error'][0], 1)
                except:
                    print("Not depolarizing: "+str(j)+" "+str(k))

    # ERROR COMPOSITION
    comb_errors_u1 = []
    comb_errors_u2 = []
    comb_errors_u3 = []
    comb_errors_cx = dict()

    # Compose errors if thermal and depolarizing are enabled
    if (err_flags[0] == 1) & (err_flags[2] == 1):
        for j in range(5):
            comb_errors_u1.append(dpol_u1[j].compose(errors_u1[j]))
            comb_errors_u2.append(dpol_u2[j].compose(errors_u2[j]))
            comb_errors_u3.append(dpol_u3[j].compose(errors_u3[j]))
            for k in range(5):
                try:
                    comb_errors_cx[j,k] = dpol_cx[j,k].compose(errors_cx[j][k])
                except:
                    print("Not composing: "+str(j)+" "+str(k))

    # ADD ERRORS TO NOISE MODEL
    noise_model = NoiseModel()
    for j in range(5):
        # Readout is enabled
        if err_flags[1] == 1:
            noise_model.add_readout_error(readout[j], [j])

        # Thermal is enabled
        if err_flags[0] == 1:
            noise_model.add_quantum_error(errors_reset[j], "reset", [j])
            noise_model.add_quantum_error(errors_measure[j], "measure", [j])

        # Thermal is enabled and dpol is not enabled
        if (err_flags[0] == 1) & (err_flags[2] == 0):
            noise_model.add_quantum_error(errors_u1[j], "u1", [j])
            noise_model.add_quantum_error(errors_u2[j], "u2", [j])
            noise_model.add_quantum_error(errors_u3[j], "u3", [j])
            for k in range(5):
                noise_model.add_quantum_error(errors_cx[j][k], "cx", [j, k])

        # Thermal is enabled and dpol is enabled
        if (err_flags[0] == 1) & (err_flags[2] == 1):
            noise_model.add_quantum_error(comb_errors_u1[j], "u1", [j])
            noise_model.add_quantum_error(comb_errors_u2[j], "u2", [j])
            noise_model.add_quantum_error(comb_errors_u3[j], "u3", [j])
            for k in range(5):
                try: noise_model.add_quantum_error(comb_errors_cx[j,k], "cx", [j, k])
                except: print("Not adding: "+str(j)+" "+str(k))

        # Thermal is not enabled and dpol is enabled
        if (err_flags[0] == 0) & (err_flags[2] == 1):
            noise_model.add_quantum_error(dpol_u1[j], "u1", [j])
            noise_model.add_quantum_error(dpol_u2[j], "u2", [j])
            noise_model.add_quantum_error(dpol_u3[j], "u3", [j])
            for k in range(5):
                try: noise_model.add_quantum_error(errors_cx[j,k], "cx", [j, k])
                except: continue

    print(noise_model)

    # Coupling map
    device = IBMQ.get_provider().get_backend(HW)
    coupling_map = device.configuration().coupling_map

    return [noise_model, coupling_map]



# Create noise model and coupling map for noisy Aer simulations
# Create noise model and coupling map for noisy Aer simulations
def hubb_customNoise(HW, times, p0g1, p1g0, err_flags):

    T1s = times[0]
    T2s = times[1]
    time_u1 = times[2]
    time_u2 = times[3]
    time_u3 = times[4]
    time_cx = times[5]
    time_reset = times[6]
    time_measure = times[7]

    # THERMAL RELAXATION ERROR
    if err_flags[0] == 1:
        errors_reset = [thermal_relaxation_error(t1, t2, time_reset) for t1, t2 in zip(T1s, T2s)]
        errors_measure = [thermal_relaxation_error(t1, t2, time_measure) for t1, t2 in zip(T1s, T2s)]
        errors_u1 = [thermal_relaxation_error(t1, t2, time_u1) for t1, t2 in zip(T1s, T2s)]
        errors_u2 = [thermal_relaxation_error(t1, t2, time_u2) for t1, t2 in zip(T1s, T2s)]
        errors_u3 = [thermal_relaxation_error(t1, t2, time_u3) for t1, t2 in zip(T1s, T2s)]
        errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(thermal_relaxation_error(t1b, t2b, time_cx))
                      for t1a, t2a in zip(T1s, T2s)]
                      for t1b, t2b in zip(T1s, T2s)]

    # READOUT ERROR
    if err_flags[1] == 1:
        readout = []
        for r in range(5):
            readout.append(ReadoutError([[1 - p1g0[r], p1g0[r]], [p0g1[r], 1 - p0g1[r]]]))

    # ADD ERRORS TO NOISE MODEL
    noise_model = NoiseModel()

    for j in range(5):
        # Readout is enabled
        if err_flags[1] == 1:
            noise_model.add_readout_error(readout[j], [j])

        # Thermal is enabled
        if err_flags[0] == 1:
            noise_model.add_quantum_error(errors_reset[j], "reset", [j])
            noise_model.add_quantum_error(errors_measure[j], "measure", [j])
            noise_model.add_quantum_error(errors_u1[j], "u1", [j])
            noise_model.add_quantum_error(errors_u2[j], "u2", [j])
            noise_model.add_quantum_error(errors_u3[j], "u3", [j])
            for k in range(5):
                noise_model.add_quantum_error(errors_cx[j][k], "cx", [j, k])

    print(noise_model)

    # Coupling map
    device = IBMQ.get_provider().get_backend(HW)
    coupling_map = device.configuration().coupling_map

    return noise_model


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
