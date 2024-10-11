from qiskit import QuantumCircuit
from qiskit import execute, Aer
import matplotlib.pyplot as plt
import numpy as np
from evovaq.problem import Problem
from evovaq.GeneticAlgorithm import GA
from evovaq.HillClimbing import HC
from evovaq.MemeticAlgorithm import MA
import evovaq.tools.operators as op
from sklearn.metrics import log_loss

    

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




    