from sklearn.decomposition import PCA
from qiskit import QuantumCircuit, execute, Aer
from qiskit.primitives import Estimator
from qiskit.quantum_info import Pauli
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from evovaq.problem import Problem
from evovaq.GeneticAlgorithm import GA
from evovaq.HillClimbing import HC
from evovaq.MemeticAlgorithm import MA
import evovaq.tools.operators as op
import pywt
from PQC_embedding import *


# Quantum Latent PCA Autoencoding Model: QLPAM
if __name__ == "__main__":
    
    df = pd.read_excel("./data/SWaT_Dataset_Attack_v0.xlsx")

    df.columns = [
        'TIMESTAMP','FIT101','LIT101','MV101','P101','P102','AIT201','AIT202','AIT203','FIT201','MV201','P201','P202','P203',
        'P204','P205','P206','DPIT301','FIT301','LIT301','MV301','MV302','MV303','MV304','P301','P302','AIT401','AIT402','FIT401',
        'LIT401','P401','P402','P403','P404','UV401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503','FIT504',
        'P501','P502','PIT501','PIT502','PIT503','FIT601','P601','P602','P603','OUTCOME'
    ]

    # The first row only contains labels
    # 'TIMESTAMP' is irrelevant for the thesis
    # The other dropped columns contain either only 0s, only 1s or only 2s and are therefore irrelevant

    df = df.iloc[1:]
    df = df.drop(['TIMESTAMP', 'P202', 'P301', 'P401', 'P404', 'P502', 'P601', 'P603'], axis = 1)

    # The dataset labels attacks by 'A ttack' and 'Attack', and labels normal data as 'Normal'
    # To keep the same structure in all datasets, the 'A ttack' and 'Attack' values are changed to '-1' and normal values to '1'

    df['OUTCOME'].replace(to_replace = ['A ttack', 'Attack'], value = 1, inplace = True)
    df['OUTCOME'].replace(to_replace = ['Normal'], value = 0, inplace = True)

    # data types need to be numeric to be encoded to z-scores --> convert column object data types to numerics

    cols = df.columns[df.columns != 'OUTCOME']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    # Encoding the feature vectors to z-scores

    cols = list(df.columns[df.columns != 'OUTCOME'])
    for col in cols:
        df[col] = ((df[col] - df[col].mean())/df[col].std(ddof=0))
        
    # Creating normal and attack masks
    normal_mask = df[df.OUTCOME == 0]
    attack_mask = df[df.OUTCOME == 1]


    
    # Creating X_training and X_testing datasets
    X_train, X_test = train_test_split(df, test_size = 0.2, random_state = 42)

    X_train = X_train[X_train.OUTCOME == 0]
    X_train = X_train.drop(['OUTCOME'], axis=1)

    y_test = X_test['OUTCOME']
    X_test = X_test.drop(['OUTCOME'], axis=1)

    X_train = X_train.values
    X_test = X_test.values

    print(X_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    def calculate_expectation_value(circuit, features, params):
        bound_circuit = circuit.bind_parameters(np.concatenate((features, params)))
        observables = [Pauli(f'Z{"I"*(len(features)-i-1)}{"I"*i}') for i in range(len(features))] 
        estimator = Estimator()
        expectation_values = [estimator.run([bound_circuit], [obs]).result().values[0] for obs in observables]
        return expectation_values

    #DA PROVARE
    def calculate_expectation_value_pqc_paper(circuit, features, params):
        pqc = costruct_embeddings_1(circuit.num_qubits, features, params)
        observables = [Pauli(f'Z{"I"*(len(features)-i-1)}{"I"*i}') for i in range(len(features))] 
        estimator = Estimator()
        expectation_values = [estimator.run([pqc], [obs]).result().values[0] for obs in observables]
        return expectation_values
        


    pca = PCA(n_components=4)
    reduced_data = pca.fit_transform(X_train)
    print(reduced_data.shape)
    dim = reduced_data.shape[1]  # LATENT SPACE PQC
            
    feature_map = ZZFeatureMap(dim, reps=1, entanglement='linear')

    # Define an Ansatz to be trained
    ansatz = RealAmplitudes(num_qubits=dim, reps=0, entanglement='circular')


    # Put together our quantum classifier
    circuit = feature_map.compose(ansatz)

    # Measure all the qubits to retrieve label information
    #circuit.measure_all()


    def mse(X, X_pred):
            return np.mean((X - X_pred) ** 2)
        
    def cost_function(params):
        expectation_values_all_samples = []
        recostruction = []
        
        for features in reduced_data:
            recostruction=calculate_expectation_value(circuit, features, params)
            expectation_values_all_samples.append(recostruction)
            
        data_reconstructed = pca.inverse_transform(expectation_values_all_samples)
        mean_s_e = mse(X_train, data_reconstructed)
        return mean_s_e  # RECOSTRUCTION LOSS

    problem = Problem(ansatz.num_parameters, ansatz.parameter_bounds, cost_function)


    # Define the global search method
    global_search = GA(selection=op.sel_tournament, crossover=op.cx_uniform, mutation=op.mut_gaussian, sigma=0.2, mut_indpb=0.15,
                cxpb=0.9, tournsize=5)

    # Create a neighbour of a possibile solution
    def get_neighbour(problem, current_solution):
        neighbour = current_solution.copy()
        index = np.random.randint(0, len(current_solution))
        _min, _max = problem.param_bounds[0]
        neighbour[index] = np.random.uniform(_min, _max)
        return neighbour

    # Define the local search method
    local_search = HC(generate_neighbour=get_neighbour)

    # Compose the global and local search method for a Memetic Algorithm 
    optimizer = MA(global_search=global_search.evolve_population, sel_for_refinement=op.sel_best, local_search=local_search.stochastic_var, frequency=0.1, intensity=10)

    res = optimizer.optimize(problem, 10, max_gen=10, verbose=True, seed=42)
    print(res)

    pca_test = PCA(n_components=4)
    pca_test_reduced = pca_test.fit_trasform(X_test)
    expectation_values_all_samples = []
    for features in pca_test_reduced:
        recostruction=calculate_expectation_value(circuit, features, res.x)
        expectation_values_all_samples.append(recostruction)
    plt.figure(figsize=(10, 6))
    sns.heatmap(expectation_values_all_samples, cmap='coolwarm', cbar_kws={'label': 'Expectation Value'})
    plt.title('Heatmap of Expectation Values per Qubit across Data Samples')
    plt.xlabel('Qubits')
    plt.ylabel('Data Samples')
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=['q0', 'q1', 'q2', 'q3'])  # Labels for each qubit
    plt.savefig('graphics/expectation_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    data_test_reconstructed = pca_test.inverse_transform(expectation_values_all_samples)
    print("MSE test: ", mse(X_test, data_test_reconstructed))
    
