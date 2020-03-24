# OPTIMIZED CIRCUIT (no sigma_z in hopping term)

# Quantum subcircuit
q = QuantumRegister(4)
hubb = QuantumCircuit(q)

pi = np.pi;
t_delta = 0.5;
v_delta = 0.25;


# Hopping 1
hubb.ry(-pi/2,1);
hubb.ry(-pi/2,3);
hubb.cx(1,3);
hubb.rz(T*t_delta*2*delta,3);
hubb.cx(1,3);
hubb.ry(pi/2,3);
hubb.ry(pi/2,1);

hubb.rx(-pi/2,1);
hubb.rx(-pi/2,3);
hubb.cx(1,3);
hubb.rz(T*t_delta*2*delta,3);
hubb.cx(1,3);
hubb.rx(pi/2,3);
hubb.rx(pi/2,1);

# Hopping 2
hubb.ry(-pi/2,0);
hubb.ry(-pi/2,2);
hubb.cx(0,2);
hubb.rz(T*t_delta*2*delta,2);
hubb.cx(0,2);
hubb.ry(pi/2,2);
hubb.ry(pi/2,0);

hubb.rx(-pi/2,0);
hubb.rx(-pi/2,2);
hubb.cx(0,2);
hubb.rz(T*t_delta*2*delta,2);
hubb.cx(0,2);
hubb.rx(pi/2,2);
hubb.rx(pi/2,0);

# Interaction 1
hubb.cx(2,3);
hubb.rz(V*v_delta*2*delta,3);
hubb.cx(2,3);

# Interaction 2
hubb.cx(0,1);
hubb.rz(V*v_delta*2*delta,1);
hubb.cx(0,1);

# Interaction 3, 4, 5, 6
hubb.rz(V*v_delta*2*delta,0);
hubb.rz(V*v_delta*2*delta,1);
hubb.rz(V*v_delta*2*delta,2);
hubb.rz(V*v_delta*2*delta,3);

################################

# Append trotterized subcircuits
for i in range(nTrot):
    hubb_fin.append(hubb, [q_fin[0], q_fin[1], q_fin[2], q_fin[3]])

# Measure final output
#hubb_fin.measure(0,0)
#hubb_fin.measure(1,1)
#hubb_fin.measure(2,2)
hubb_fin.measure(3,3)
