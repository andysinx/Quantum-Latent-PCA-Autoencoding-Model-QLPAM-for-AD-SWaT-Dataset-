from sklearn.decomposition import PCA
from qiskit_aer import AerError
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2
from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, confusion_matrix, roc_curve, recall_score, 
                             classification_report, roc_auc_score, f1_score, precision_recall_fscore_support)
from evovaq.problem import Problem
from evovaq.GeneticAlgorithm import GA
from evovaq.HillClimbing import HC
from evovaq.MemeticAlgorithm import MA
import evovaq.tools.operators as op
from fastdtw import fastdtw
from scipy.spatial.distance import pdist, squareform



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
    
    
    sensors=['FIT101','LIT101']
    sampled_data = df.sample(n=9000, random_state=42)
    sampled_data = sampled_data.astype(np.float64)
    sampled_data_dtw=sampled_data[sensors]
    
    
    def compute_dtw_matrix(X):
        num_cores = -1
        X_array = X.values
        dtw_distances = squareform(pdist(X_array.T, metric=lambda u, v: fastdtw(u, v)[0]))
        return dtw_distances

    #dtw_distances = compute_dtw_matrix(sampled_data)
    #sampled_data['dtw_distances'] = dtw_distances
    
    # Creating X_training and X_testing datasets
    X_train, X_test = train_test_split(sampled_data, test_size = 0.2, random_state = 42)

    X_train = X_train[X_train.OUTCOME == 0]
    y_train = X_train['OUTCOME']
    X_train = X_train.drop(['OUTCOME'], axis=1)

    y_test = X_test['OUTCOME']
    X_test = X_test.drop(['OUTCOME'], axis=1)

    X_train = X_train.values
    X_test = X_test.values
    
    print(X_train.shape)
    print(X_test.shape)
    print(y_test.shape)


    def calculate_expectation_value(circuit, features, params):
        try:
            gpu_estimator = AerSimulator(method='statevector', device='GPU')
            gpu_estimator.set_options(precision='single')
        except AerError as e:
            print(e)    
        print("feature:", features)
        print("params:", params)
        bound_circuit = circuit.assign_parameters(np.concatenate((features, params)))
        observables = [Pauli('ZIII'), Pauli('IZII'), Pauli('IIZI'), Pauli('IIIZ')]
        estimator = EstimatorV2(gpu_estimator)
        expectation_values = [estimator.run([(bound_circuit, obs)]).result()[0].data.evs for obs in observables]
        print("expectation_values: ", expectation_values)
        exp_val = [exp.item() for exp in expectation_values]
        print("exp_values: ", exp_val)
        return exp_val
        
    pca = PCA(n_components=4)
    reduced_data = pca.fit_transform(X_train)
    print(reduced_data.shape)
    dim = reduced_data.shape[1]  # LATENT SPACE PQC
            
    ansatz = TwoLocal(dim, rotation_blocks=['rx', 'ry', 'rz'], entanglement_blocks=['cx', 'swap', 'h'], entanglement='circular', reps=2 , insert_barriers=True)

    # Define an Ansatz to be trained
    feature_map = QuantumCircuit(dim)
    feature_map &= ansatz
    feature_map = feature_map.decompose()
    feature_map.draw(output='mpl')
    plt.savefig('graphics/quantum_kernels.png')


    # Put together our quantum classifier
    circuit = feature_map

    # Measure all the qubits to retrieve label information
    #circuit.measure_all()

    iso_model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.012),
                    max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)

        
    def cost_function(params):
        expectation_values_all_samples = []
        recostruction = []
        
        for features in reduced_data:
            recostruction=calculate_expectation_value(circuit, features, params)
            expectation_values_all_samples.append(recostruction)
        mid_index = len(expectation_values_all_samples) // 2
        data_train_iso = expectation_values_all_samples[:mid_index]
        data_val_iso = expectation_values_all_samples[mid_index:]
        #data_reconstructed = pca.inverse_transform(expectation_values_all_samples)
        print("data_train_iso: ", data_train_iso)
        print("data_val_iso: ", data_val_iso)
        iso_model.fit(data_train_iso)
        iso_predictions = iso_model.predict(data_val_iso)
        iso_predictions = [1 if pred == -1 else 0 for pred in iso_predictions]
        iso_predictions = np.array(iso_predictions, dtype=float)
        print("iso_predictions: ", iso_predictions)
        iso_true = y_train[mid_index:].values
        accuracy = accuracy_score(iso_true, iso_predictions)
        print("iso_true: ", iso_true)
        print("accuracy: ", accuracy)
        return accuracy  # RECOSTRUCTION LOSS

    
    problem = Problem(n_params=ansatz.num_parameters-4, param_bounds=(-np.pi,np.pi), obj_function=cost_function)


    # Define the global search method
    global_search = GA(selection=op.sel_tournament, crossover=op.cx_uniform, mutation=op.mut_gaussian, sigma=0.2, mut_indpb=0.15,
                cxpb=0.9, tournsize=5)

    # Create a neighbour of a possibile solution
    def get_neighbour(problem, current_solution):
        neighbour = current_solution.copy()
        index = np.random.randint(0, len(current_solution))
        print("problem.param_bounds: ", problem.param_bounds)
        _min, _max = problem.param_bounds
        neighbour[index] = np.random.uniform(_min, _max)
        return neighbour

    # Define the local search method
    local_search = HC(generate_neighbour=get_neighbour)

    # Compose the global and local search method for a Memetic Algorithm 
    optimizer = MA(global_search=global_search.evolve_population, sel_for_refinement=op.sel_best, local_search=local_search.stochastic_var, frequency=0.1, intensity=10)

    res = optimizer.optimize(problem, 10, max_gen=10, verbose=True, seed=42)
    print(res)
    
    pca = PCA(n_components=4)
    pca_test_reduced = pca.fit_transform(X_test)
    expectation_values_all_samples = []
    for features in pca_test_reduced:
        recostruction=calculate_expectation_value(circuit, features, res.x)
        expectation_values_all_samples.append(recostruction)
    iso_predictions = iso_model.predict(expectation_values_all_samples)
    iso_binary_predictions = [1 if pred == -1 else 0 for pred in iso_predictions]
    iso_true_labels = y_test
    iso_predicted_labels = iso_binary_predictions
    
    #metrics test set
    accuracy = accuracy_score(iso_true_labels, iso_binary_predictions)
    conf_matrix = confusion_matrix(iso_true_labels, iso_binary_predictions)
    roc_auc = roc_auc_score(iso_true_labels, iso_binary_predictions)
    fpr, tpr, _ = roc_curve(iso_true_labels, iso_binary_predictions)
    recall = recall_score(iso_true_labels, iso_binary_predictions)
    precision = precision_score(iso_true_labels, iso_binary_predictions)
    f1 = f1_score(iso_true_labels, iso_binary_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(iso_true_labels, iso_binary_predictions)
    
    
    # Creazione della tabella delle metriche
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'ROC AUC Score', 'Recall', 'Precision', 'F1 Score'],
        'Value': [accuracy, roc_auc, recall, precision, f1]
    })

    print("Metriche di Valutazione:")
    print(metrics_df)
    print("\nClassification Report:\n", classification_report(iso_true_labels, iso_binary_predictions))
    
    # Stampa della matrice di confusione
    LABELS = ['Normal', 'Anomaly']
    print("Confusion Matrix:\n", conf_matrix)
    plt.figure(figsize=(12, 12))
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.savefig('graphics/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Grafico della curva ROC
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Linea diagonale
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('graphics/roc_curve_isolation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(expectation_values_all_samples, cmap='coolwarm', cbar_kws={'label': 'Expectation Value'})
    plt.title('Heatmap of Expectation Values per Qubit across Data Samples')
    plt.xlabel('Qubits')
    plt.ylabel('Data Samples')
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=['q0', 'q1', 'q2', 'q3'])  # Etichetta per ogni qubit
    plt.savefig('graphics/expectation_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
