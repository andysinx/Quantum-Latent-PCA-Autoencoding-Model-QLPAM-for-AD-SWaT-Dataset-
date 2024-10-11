from qiskit import QuantumCircuit

    

def costruct_embeddings_1(num_qubits,features,params):
    pqc= QuantumCircuit(num_qubits)
    for qubit in range(num_qubits):
        pqc.rx(features[qubit], qubit)
            
    for qubit in range(num_qubits):
        pqc.h(qubit)
        
    pqc.swap(0, 3)
    pqc.swap(0, 1)
    pqc.swap(1, 2)
    pqc.swap(2, 3)
    
    for qubit in range(num_qubits):
        pqc.rx(params[qubit], qubit)
    return pqc 




    
