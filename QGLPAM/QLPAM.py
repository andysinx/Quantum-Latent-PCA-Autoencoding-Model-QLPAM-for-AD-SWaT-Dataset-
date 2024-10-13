import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import SPSA
from qiskit.primitives import Estimator
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data():
    logging.info("Loading data from SWaT_Dataset_Attack_v0_reduced.xlsx.")
    df = pd.read_excel("../resources/SWaT_Dataset_Attack_v0.xlsx")
    logging.info("Data loaded successfully. Preprocessing...")

    df.columns = [
        'TIMESTAMP', 'FIT101', 'LIT101', 'MV101', 'P101', 'P102', 'AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201',
        'P201', 'P202', 'P203',
        'P204', 'P205', 'P206', 'DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302',
        'AIT401', 'AIT402', 'FIT401',
        'LIT401', 'P401', 'P402', 'P403', 'P404', 'UV401', 'AIT501', 'AIT502', 'AIT503', 'AIT504', 'FIT501', 'FIT502',
        'FIT503', 'FIT504',
        'P501', 'P502', 'PIT501', 'PIT502', 'PIT503', 'FIT601', 'P601', 'P602', 'P603', 'OUTCOME'
    ]
    # The first row only contains labels
    # 'TIMESTAMP' is irrelevant for the thesis
    # The other dropped columns contain either only 0s, only 1s or only 2s and are therefore irrelevant

    df = df.iloc[1:]
    df = df.drop(['TIMESTAMP', 'P202', 'P301', 'P401', 'P404', 'P502', 'P601', 'P603'], axis=1)

    # The dataset labels attacks by 'A ttack' and 'Attack', and labels normal data as 'Normal'
    # To keep the same structure in all datasets, the 'A ttack' and 'Attack' values are changed to '-1' and normal values to '1'

    df['OUTCOME'].replace(to_replace=['A ttack', 'Attack'], value=1, inplace=True)
    df['OUTCOME'].replace(to_replace=['Normal'], value=0, inplace=True)
    # data types need to be numeric to be encoded to z-scores --> convert column object data types to numerics
    cols = df.columns[df.columns != 'OUTCOME']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    # Encoding the feature vectors to z-scores
    cols = list(df.columns[df.columns != 'OUTCOME'])
    for col in cols:
        df[col] = ((df[col] - df[col].mean()) / df[col].std(ddof=0))

    # Creating X_training and X_testing datasets
    X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)
    X_train = X_train[X_train.OUTCOME == 0]
    X_train = X_train.drop(['OUTCOME'], axis=1)
    y_test = X_test['OUTCOME']
    X_test = X_test.drop(['OUTCOME'], axis=1)

    X_train = X_train.values
    X_test = X_test.values

    logging.info(f"Training data shape: {X_train.shape}")
    logging.info(f"Testing data shape: {X_test.shape}")
    logging.info(f"Test labels shape: {y_test.shape}")

    return X_train, X_test, y_test


def construct_pqc(num_qubits, features, params):
    pqc = QuantumCircuit(num_qubits)
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


# Function to calculate the expectation value
def calculate_expectation_value(num_qubits, features, params):
    # Copy the base circuit so that it's not modified
    pqc = construct_pqc(num_qubits, features, params)

    # Define Pauli observables
    observables = [Pauli(f'{"I" * i}Z{"I" * (num_qubits - i - 1)}') for i in range(num_qubits)]
    # observables = [Pauli(f'Z{"I" * (len(features) - i - 1)}{"I" * i}') for i in range(len(features))] # Andrea

    estimator = Estimator()

    # Calculate the expectation value for each observable
    expectation_values = [estimator.run([pqc], [obs]).result().values[0] for obs in observables]
    return expectation_values


def mse(X, X_pred):
    return np.mean((X - X_pred) ** 2)


def cost_function(params, reduced_data, X_train, pca, dim_cf):
    logging.info("Starting cost function evaluation.")
    local_expectation_values_all_samples = []

    # Loop over each sample and calculate the expectation values dynamically
    for i, local_features in enumerate(reduced_data):
        logging.info(f"Calculating expectation value for sample {i + 1}/{len(reduced_data)}.")
        local_reconstruction = calculate_expectation_value(dim_cf, local_features, params)
        local_expectation_values_all_samples.append(local_reconstruction)

    logging.info("All expectation values calculated. Reconstructing data from expectation values.")
    # Perform inverse PCA transform to get back to original space
    local_data_reconstructed = pca.inverse_transform(local_expectation_values_all_samples)

    # Calculate and return the Mean Squared Error
    mean_s_e = mse(X_train, local_data_reconstructed)
    logging.info(f"Mean squared error (MSE) calculated: {mean_s_e:.6f}.")
    return mean_s_e  # Reconstruction loss


# QAOA optimization routine
def optimize_params_qaoa(local_reduced_data, n_qubits, X_train_set, pc_analysis):
    initial_params = np.random.uniform(0, 2 * np.pi, n_qubits)  # Initial guess for parameters

    # Optimization using a classical optimizer (COBYLA)
    # result = minimize(cost_function, initial_params,
    #                   args=(local_reduced_data, X_train_set, pc_analysis, dim_cf, base_pqc),
    #                   method='COBYLA')

    # Define the wrapper function inside optimize_params_qaoa
    def wrapped_cost_function(params):
        return cost_function(params, local_reduced_data, X_train_set, pc_analysis, n_qubits)

    # SPSA optimizer (suitable for quantum circuits)
    spsa = SPSA(maxiter=100)  # Adjust maxiter as per performance
    # Minimize the wrapped cost function
    result = spsa.minimize(wrapped_cost_function, initial_params)

    # Get the optimal parameters
    opt_params = result.x
    return opt_params


def test(test_set, params, original_pca):
    pca_test_reduced = original_pca.transform(test_set)

    n_qubits = pca_test_reduced.shape[1]

    local_expectation_values_all_samples = []
    for local_features in pca_test_reduced:
        local_reconstruction = calculate_expectation_value(local_features, params, n_qubits)
        local_expectation_values_all_samples.append(local_reconstruction)

    plt.figure(figsize=(10, 6))
    sns.heatmap(local_expectation_values_all_samples, cmap='coolwarm', cbar_kws={'label': 'Expectation Value'})
    plt.title('Heatmap of Expectation Values per Qubit across Data Samples')
    plt.xlabel('Qubits')
    plt.ylabel('Data Samples')
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=['q0', 'q1', 'q2', 'q3'])  # Labels for each qubit
    plt.savefig('../graphics/expectation_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    data_test_reconstructed = original_pca.inverse_transform(local_expectation_values_all_samples)
    print("MSE test: ", mse(test_set, data_test_reconstructed))


# Quantum Latent PCA Autoencoding Model with EVOVAQ Optimizer: QGLPAM
if __name__ == "__main__":
    logging.info("Starting the quantum latent PCA autoencoding model.")
    X_train, X_test, y_test = load_data()

    X_train = X_train[:-270000, :]
    X_test = X_test[:-76800, :]
    y_test = y_test[:-76800]

    logging.info("Applying PCA to reduce dimensionality of training data.")
    pca = PCA(n_components=4)

    reduced_data = pca.fit_transform(X_train)

    logging.info(f"PCA reduced data shape: {reduced_data.shape}.")
    dim = reduced_data.shape[1]  # LATENT SPACE PQC

    num_qubits = dim

    logging.info("Optimizing parameters using QAOA.")
    # Optimize the parameters using QAOA
    optimal_params = optimize_params_qaoa(reduced_data, num_qubits, X_train, pca, dim)

    logging.info("Optimal parameters found by QAOA:")
    logging.info(f"{optimal_params}")

    # Optionally, test on the test set
    test(X_test, optimal_params, pca)
