import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from qiskit_aer import AerError
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2
from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from keras.src import Sequential
from keras.src.layers import Dense
from keras.src.callbacks import ModelCheckpoint, TensorBoard
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from HQGA import problems, hqga_algorithm, hqga_utils, utils
from sklearn.metrics import (accuracy_score, precision_score, confusion_matrix, roc_curve, recall_score,
                             classification_report, roc_auc_score, f1_score, precision_recall_fscore_support)
# from fastdtw import fastdtw
from scipy.spatial.distance import pdist, squareform


def load_data():
    df = pd.read_excel("./data/SWaT_Dataset_Attack_v0.xlsx")

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

    # Creating normal and attack masks
    normal_mask = df[df.OUTCOME == 0]
    attack_mask = df[df.OUTCOME == 1]

    sensors = ['FIT101', 'LIT101']
    sampled_data = df.sample(n=5000, random_state=42)
    sampled_data = sampled_data.astype(np.float64)
    sampled_data_dtw = sampled_data[sensors]

    # Creating X_training and X_testing datasets
    X_train, X_test = train_test_split(sampled_data, test_size=0.2, random_state=42)

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

    return X_train, X_test, y_test, y_train


def get_reduced_data_with_nn(X_train):
    # PCA O ENCODING NEURALE (BASTA COMMENTARE UNO E SCOMMENTARE L'ALTRO)
    ae = model = Sequential()
    model.add(Dense(35, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(15, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(35, activation='relu'))
    model.add(Dense(X_train.shape[1]))

    nb_epoch = 150
    batch_size = 64

    ae.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath="./model.SWAT.keras", verbose=0)

    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

    history = ae.fit(X_train, X_train, epochs=nb_epoch, batch_size=batch_size, validation_data=(X_test, X_test),
                     verbose=1, callbacks=[checkpointer, tensorboard]).history

    encoder_model = Sequential()  # Modello separato per ottenere solo la parte di encoding
    encoder_model.add(ae.layers[0])  # Strato di input
    encoder_model.add(ae.layers[1])
    encoder_model.add(ae.layers[2])
    encoder_model.add(ae.layers[3])
    encoder_model.add(ae.layers[4])
    encoder_model.add(ae.layers[5])

    return encoder_model.predict(X_train)


def calculate_expectation_value(circuit, features, params):
    try:
        gpu_estimator = AerSimulator(method='statevector', device='GPU')
        gpu_estimator.set_options(precision='single')
    except AerError as e:
        print(e)
    bound_circuit = circuit.assign_parameters(np.concatenate((features, params)))
    observables = [Pauli('ZIII'), Pauli('IZII'), Pauli('IIZI'), Pauli('IIIZ')]
    estimator = EstimatorV2(gpu_estimator)
    expectation_values = [estimator.run([(bound_circuit, obs)]).result()[0].data.evs for obs in observables]
    exp_val = [exp.item() for exp in expectation_values]
    return exp_val


def test(pca, X_test, y_test, iso_model, res, circuit):
    pca_test_reduced = pca.transform(X_test)
    expectation_values_all_samples = []

    for features in pca_test_reduced:
        reconstruction = calculate_expectation_value(circuit, features, res)
        expectation_values_all_samples.append(reconstruction)

    iso_predictions = iso_model.predict(expectation_values_all_samples)
    iso_binary_predictions = [1 if pred == -1 else 0 for pred in iso_predictions]
    iso_true_labels = y_test
    iso_predicted_labels = iso_binary_predictions

    # metrics test set
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
    plt.savefig('graphics_1/confusion_matrix_test.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('graphics_1/roc_curve_isolation.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(expectation_values_all_samples, cmap='coolwarm', cbar_kws={'label': 'Expectation Value'})
    plt.title('Heatmap of Expectation Values per Qubit across Data Samples')
    plt.xlabel('Qubits')
    plt.ylabel('Data Samples')
    plt.xticks(ticks=[0.5, 1.5, 2.5, 3.5], labels=['q0', 'q1', 'q2', 'q3'])  # Etichetta per ogni qubit
    plt.savefig('graphics_1/expectation_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # data_test_reconstructed = pca.inverse_transform(expectation_values_all_samples)
    # print("MSE test: ", mse(X_test, data_test_reconstructed))


def main():
    X_train, X_test, y_test, y_train = load_data()

    # Reduced Data with Neural Network
    # reduced_data = get_reduced_data_with_nn(X_train)

    # Reduced Data with PCA
    pca = PCA(n_components=4)
    reduced_data = pca.fit_transform(X_train)

    dim = reduced_data.shape[1]  # LATENT SPACE PQC

    ansatz = TwoLocal(dim, rotation_blocks=['rx', 'ry', 'rz'], entanglement_blocks=['cx', 'swap', 'h'],
                      entanglement='circular', reps=1, insert_barriers=True)

    # Define an Ansatz to be trained
    feature_map = QuantumCircuit(dim)
    feature_map &= ansatz
    feature_map = feature_map.decompose()
    feature_map.draw(output='mpl')
    plt.savefig('graphics/quantum_kernels.png')

    # Put together our quantum classifier
    circuit = feature_map

    # Measure all the qubits to retrieve label information
    # circuit.measure_all()

    real_quantum_hardware = False

    if real_quantum_hardware:
        # WRITE CODE
        backend = ""  # COMPLETE CODE
    else:
        try:
            backend = AerSimulator(method='tensor_network', device='GPU')
            backend.set_options(precision='single')
        except AerError as e:
            print(e)
    device_features = hqga_utils.device(backend)

    # hyper-parameters in the case of the quantum elitism
    # k = 20
    k = 3
    # max_gen = 50
    max_gen = 5
    delta = np.pi / 8
    mu = 0.3
    rho = np.pi / 16
    elitism = hqga_utils.ELITISM_Q

    if elitism == hqga_utils.ELITISM_Q or elitism == hqga_utils.ELITISM_D:
        params = hqga_utils.Parameters(k, max_gen, delta, mu, elitism)
    elif elitism == hqga_utils.ELITISM_R:
        params = hqga_utils.ReinforcementParameters(k, max_gen, delta, rho, mu)
    else:
        print("Please, select one elitism procedure among ELITISM_Q, ELITISM_D and ELITISM_R")

    # presentation hyper-parameters
    params.progressBar = True
    params.verbose = True
    params.draw_circuit = True

    iso_model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.012),
                                max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)

    problem = problems.VariationalProblem(4, 5, -np.pi, np.pi, circuit, iso_model, reduced_data, y_train)

    gBest, chromosome_evolution, bests = hqga_algorithm.runQGA(device_features, circuit, params, problem)

    print("Best solution", gBest.chr)
    print("Optimal solutions", problem.getOptimum()[2])
    for each_sol in problem.getOptimum()[2]:
        dis = utils.hamming_distance(gBest.chr, each_sol[0])

        print("Hamming distance", dis)

    test(pca, X_test, y_test, iso_model, gBest, circuit)


# Quantum Latent PCA Autoencoding Model: QLPAM
if __name__ == "__main__":
    main()
