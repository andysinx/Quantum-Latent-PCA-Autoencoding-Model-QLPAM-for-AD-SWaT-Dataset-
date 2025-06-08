import evovaq.tools.operators as op
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evovaq.GeneticAlgorithm import GA
from evovaq.HillClimbing import HC
from evovaq.MemeticAlgorithm import MA
from evovaq.problem import Problem
from keras.src.callbacks import ModelCheckpoint, TensorBoard
from keras.src.layers import Dense
from keras.src.models import Sequential
from qiskit import QuantumCircuit
from qiskit.circuit.library import TwoLocal
from qiskit.quantum_info import Pauli
from qiskit_aer import AerError
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (roc_curve, roc_auc_score, accuracy_score, auc)
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM



# Quantum Latent PCA Autoencoding Model: QLPAM
if __name__ == "__main__":

    df = pd.read_csv("./data/SWaT_Dataset_Attack_v0.csv")

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


    # 🔹 Definisci feature e target
    X = df.drop(columns=["OUTCOME"])  # Tutte le colonne tranne il target
    y = df["OUTCOME"]  # Target binario (0 = normale, 1 = anomalia)

    # 🔹 Calcola la Mutual Information
    mi_scores = mutual_info_classif(X, y)

    # 🔹 Crea un DataFrame con i risultati
    mi_results = pd.DataFrame({"Feature": X.columns, "MI_Score": mi_scores})

    # 🔹 Filtra solo le feature con MI > 0 (cioè, che portano informazione)
    threshold = 0.2098
    selected_features = mi_results[mi_results["MI_Score"] > threshold]

    # 🔹 Ordina in base all'importanza
    selected_features = selected_features.sort_values(by="MI_Score", ascending=False)

    # 🔹 Stampa le feature selezionate
    print("Feature più informative rispetto alle anomalie:")
    print(selected_features)

    # 🔹 Se vuoi solo i nomi delle feature per usarle in MATLAB
    important_feature_names = selected_features["Feature"].tolist()
    print("\nFeature selezionate:", important_feature_names)

    # Supponiamo che `df` sia il tuo DataFrame originale
    selected_features = ['AIT201', 'AIT501', 'UV401', 'P501']

    # Creiamo un nuovo DataFrame con solo queste feature + il target
    df_selected = df[selected_features]

    df_moving_avg = df_selected.rolling(window=5).mean()
    df_moving_avg = df_moving_avg.dropna()
    # Plottiamo ogni colonna in una figura separata
    #i = 0
    #for col in df_selected.columns:
    #    plt.figure(figsize=(12, 6))
    #    plt.plot(df.index, df_selected[col], label=f'Original {col}', color='blue')
    #    plt.plot(df_moving_avg.index, df_moving_avg[col], label=f'Moving Average {col}', color='red', linestyle='--')
    #    plt.xlabel('Time')
    #    plt.ylabel(col)
    #    plt.title(f'{col} and Moving Average')
    #    plt.legend()
    #    plt.savefig(f'graphics/remove_noise_lowpassfilter_{i}.png')
    #    plt.close()
    #    i +=1

    # Encoding the feature vectors to min-max

    cols = list(df_moving_avg.columns)
    scaler = MinMaxScaler()
    for col in cols:
        df_moving_avg[cols] = scaler.fit_transform(df_moving_avg[cols])

    df_selected["OUTCOME"] = df["OUTCOME"]
    df_moving_avg["OUTCOME"] = df["OUTCOME"]
    df_selected =df_selected[selected_features + ["OUTCOME"]]
    # Salviamo il file CSV
    df_selected.to_csv("./data/relevant_features.csv", index=False)

    # 🔹 Carica il dataset (se non è già in memoria)
    df = pd.read_csv("./data/relevant_features.csv")

    # 🔹 Seleziona solo le feature di input (senza OUTCOME)
    X = df.drop(columns=["OUTCOME"])
    y = df["OUTCOME"]

    # 🔹 Filtra solo dati normali per training e validation set
    df_normal = df_moving_avg[df_moving_avg["OUTCOME"] == 0]
    df_anomaly = df_moving_avg[df_moving_avg["OUTCOME"] == 1]

    # 🔹 Prendi i primi 9000 dati normali per il training set
    X_train = df_normal.iloc[:9000, :-1].values  # Escludi OUTCOME
    X_train = np.concatenate([X_train[:2000], X_train[-100:]])  # 123123
    y_train = df_normal.iloc[:9000, -1].values  # Target
    y_train = np.concatenate([y_train[:2000], y_train[-100:]]) # 123123

    # 🔹 Prendi i successivi 9000 dati normali per il validation set
    X_val = df_normal.iloc[9000:18000, :-1].values
    y_val = df_normal.iloc[9000:18000, -1].values

    # 🔹 Crea test set bilanciato (50% anomalie) - 1750 normali + 1750 anomalie
    X_test_balanced = pd.concat([df_normal.iloc[18000:19750, :-1], df_anomaly.iloc[:1750, :-1]]).values
    y_test_balanced = pd.concat([df_normal.iloc[18000:19750, -1], df_anomaly.iloc[:1750, -1]]).values

    # 🔹 Crea test set con 5% anomalie - 3325 normali + 175 anomalie
    X_test_5perc = pd.concat([df_normal.iloc[19750:23075, :-1], df_anomaly.iloc[1750:1925, :-1]]).values
    X_test_5perc = np.concatenate([X_test_5perc[:2000], X_test_5perc[-100:]])  # 123123
    y_test_5perc = pd.concat([df_normal.iloc[19750:23075, -1], df_anomaly.iloc[1750:1925, -1]]).values
    y_test_5perc = np.concatenate([y_test_5perc[:2000], y_test_5perc[-100:]])  # 123123

    X_val_5perc = pd.concat([df_normal.iloc[9000:18000, :-1], df_anomaly.iloc[1750:1925, :-1]]).values
    X_val_5perc = np.concatenate([X_val_5perc[:2000], X_val_5perc[-100:]])  # 123123
    y_val_5perc = pd.concat([df_normal.iloc[9000:18000, -1], df_anomaly.iloc[1750:1925, -1]]).values
    y_val_5perc = np.concatenate([y_val_5perc[:2000], y_val_5perc[-100:]])  # 123123


    def calculate_expectation_value(circuit, features, params):
        try:
            gpu_estimator = AerSimulator(method='statevector', device="GPU")
            gpu_estimator.set_options(precision='single')
        except AerError as e:
            gpu_estimator = None
            print(e)

        if gpu_estimator is None:
            return None
        bound_circuit = circuit.assign_parameters(np.concatenate((features, params)))
        observables = [Pauli('ZIII'), Pauli('IZII'), Pauli('IIZI'), Pauli('IIIZ')]
        estimator = EstimatorV2(gpu_estimator)
        expectation_values = [estimator.run([(bound_circuit, obs)]).result()[0].data.evs for obs in observables]
        exp_val = [exp.item() for exp in expectation_values]
        return exp_val


    # PCA O ENCODING NEURALE (BASTA COMMENTARE UNO E SCOMMENTARE L'ALTRO)
    '''ae = model = Sequential()
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

    history = ae.fit(X_train, X_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True, validation_data=(X_test, X_test), verbose=1, callbacks=[checkpointer, tensorboard]).history


    encoder_model = Sequential()  # Modello separato per ottenere solo la parte di encoding
    encoder_model.add(ae.layers[0])  # Strato di input
    encoder_model.add(ae.layers[1])
    encoder_model.add(ae.layers[2])
    encoder_model.add(ae.layers[3])
    encoder_model.add(ae.layers[4])
    encoder_model.add(ae.layers[5])

    reduced_data = encoder_model.predict(X_train)'''

    #pca = PCA(n_components=4)
    reduced_data_train = X_train.copy()
    reduced_data_val = X_val_5perc.copy()
    dim = reduced_data_train.shape[1]  # LATENT SPACE PQC

    ansatz = TwoLocal(dim, rotation_blocks=['rx', 'ry', 'rz'], entanglement_blocks=['cx','h','swap'],
                      entanglement='linear', reps=1, insert_barriers=True)

    # Define an Ansatz to be trained
    feature_map = QuantumCircuit(dim)
    feature_map &= ansatz
    feature_map = feature_map.decompose()
    feature_map.draw(output='mpl')
    plt.savefig('graphics/quantum_kernels_f2.pdf')

    # Put together our quantum classifier
    circuit = feature_map

    iso_forest = IsolationForest()
    one_class = OneClassSVM()


    def custom_scorer(estimator, X):
        decision_scores = estimator.decision_function(X)
        roc_score = roc_auc_score(y_val_5perc, decision_scores)
        return roc_score


    model = Sequential()
    model.add(Dense(3, input_dim=reduced_data_train.shape[1], activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(3, activation='relu'))
    model.add(Dense(reduced_data_train.shape[1]))


    def cost_function(params):
        global one_class
        expectation_values_all_samples = []
        expectation_values_all_samples_1 = []

        for features in reduced_data_train:
            reconstruction = calculate_expectation_value(circuit, features, params)
            expectation_values_all_samples.append(reconstruction)

        for features in reduced_data_val:
            reconstruction_1 = calculate_expectation_value(circuit, features, params)
            expectation_values_all_samples_1.append(reconstruction_1)

        kernels = ['linear', 'poly', 'sigmoid', 'rbf']
        nus = [0.01, 0.05, 0.1, 0.2, 0.5]
        gammas = ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10]
        best_params = None
        best_score = -np.inf  # Puoi settarlo anche a np.inf se cerchi il valore minimo

        # Loop manuale su tutte le combinazioni di parametri
        for kernel in kernels:
            for nu in nus:
                for gamma in gammas:
                    # Inizializza il modello con i parametri attuali
                    model_oneclass = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
                    # Fitta sul training set
                    model_oneclass.fit(expectation_values_all_samples)
                    # Calcola il punteggio sul validation set
                    score = custom_scorer(model_oneclass, expectation_values_all_samples_1)
                    # Se il punteggio attuale è migliore, aggiorna i parametri ottimali
                    if score > best_score:
                        best_score = score
                        best_params = {'kernel': kernel, 'nu': nu, 'gamma': gamma}
        one_class = OneClassSVM(**best_params)
        one_class.fit(expectation_values_all_samples)
        oc_predictions = one_class.decision_function(expectation_values_all_samples_1)
        roc_score = roc_auc_score(y_val_5perc, oc_predictions)

        return -1 * roc_score


    def cost_function_1(params):
        global iso_forest
        expectation_values_all_samples = []
        expectation_values_all_samples_1 = []

        for features in reduced_data_train:
            reconstruction = calculate_expectation_value(circuit, features, params)
            expectation_values_all_samples.append(reconstruction)

        for features in reduced_data_val:
            reconstruction_1 = calculate_expectation_value(circuit, features, params)
            expectation_values_all_samples_1.append(reconstruction_1)

        n_estimators_options = [50, 100, 150]
        max_samples_options = [0.5, 0.75, 1.0]
        max_features_options = [0.5, 0.75, 1.0]
        contamination = 160 / 1440  # Contaminazione basata sui dati forniti

        # Variabili per tracciare i migliori parametri e il miglior punteggio
        best_params = None
        best_score = -np.inf

        # Loop su tutte le combinazioni di parametri
        for n_estimators in n_estimators_options:
            for max_samples in max_samples_options:
                for max_features in max_features_options:
                    # Inizializza il modello Isolation Forest con i parametri correnti
                    model_iso = IsolationForest(
                        n_estimators=n_estimators,
                        max_samples=max_samples,
                        max_features=max_features,
                        contamination=contamination,
                        random_state=42
                    )
                    # Fitta sul training set
                    model_iso.fit(expectation_values_all_samples)
                    # Calcola il punteggio sul validation set
                    score = custom_scorer(model_iso, expectation_values_all_samples_1)
                    # Se il punteggio attuale è migliore, aggiorna i parametri ottimali
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'n_estimators': n_estimators,
                            'max_samples': max_samples,
                            'max_features': max_features
                        }

        iso_forest = IsolationForest(**best_params, contamination=contamination, random_state=42)
        iso_forest.fit(expectation_values_all_samples)
        iso_predictions = iso_forest.decision_function(expectation_values_all_samples_1)
        iso_roc_score = roc_auc_score(y_val_5perc, iso_predictions)

        return -1 * iso_roc_score


    def cost_function_2(params):
        global model
        expectation_values_all_samples = []
        expectation_values_all_samples_1 = []

        for features in reduced_data_train:
            reconstruction = calculate_expectation_value(circuit, features, params)
            expectation_values_all_samples.append(reconstruction)

        for features in reduced_data_val:
            reconstruction_1 = calculate_expectation_value(circuit, features, params)
            expectation_values_all_samples_1.append(reconstruction_1)

        nb_epoch = 25  # 123123
        batch_size = 10

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

        checkpointer = ModelCheckpoint(filepath="./model.SWAT.keras", verbose=0)

        tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)

        model.fit(
            np.array(expectation_values_all_samples),
            np.array(expectation_values_all_samples),
            epochs=nb_epoch,
            batch_size=batch_size,
            # validation_data=(np.array(expectation_values_all_samples_1), np.array(expectation_values_all_samples_1)),
            verbose=1,
            callbacks=[checkpointer, tensorboard]
        )

        lab = model.predict(np.array(expectation_values_all_samples_1))

        reconstruction_errors = np.linalg.norm(np.array(expectation_values_all_samples_1) - lab, axis=1)

        # Definire una soglia per determinare se un campione è anomalo o meno
        threshold = np.percentile(reconstruction_errors, 95)  # esempio: 95% dei campioni più normali
        y_pred = (reconstruction_errors > threshold).astype(int)  # 1 per anomalia, 0 per normale
        ae_score = auc(y_val_5perc, y_pred)
        return -1 * ae_score

    iso_model = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(.012),
                                max_features=1.0, bootstrap=False, random_state=42, verbose=0)

    #def cost_function_if(params):
    #    expectation_values_all_samples = []
    #    for features in reduced_data_train:
    #        recostruction = calculate_expectation_value(circuit, features, params)
    #        expectation_values_all_samples.append(recostruction)
    #    mid_index = len(expectation_values_all_samples) // 2
    #    data_train_iso = expectation_values_all_samples[:mid_index]
    #    data_val_iso = expectation_values_all_samples[mid_index:]
    #    iso_model.fit(data_train_iso)
    #    iso_predictions = iso_model.predict(data_val_iso)
    #    iso_predictions = np.array([1 if pred == -1 else 0 for pred in iso_predictions], dtype=float)
    #    iso_true = y_train[mid_index:]
    #    accuracy = accuracy_score(iso_true, iso_predictions)
    #    return accuracy  # RECOSTRUCTION LOSS

    problem_oneclass = Problem(
        n_params=ansatz.num_parameters - 4,
        param_bounds=(-np.pi, np.pi),
        obj_function=cost_function
    )

    problem_iso_forest = Problem(
        n_params=ansatz.num_parameters - 4,
        param_bounds=(-np.pi, np.pi),
        obj_function=cost_function_1
    )

    problem_ae = Problem(
        n_params=ansatz.num_parameters - 4,
        param_bounds=(-np.pi, np.pi),
        obj_function=cost_function_2
    )

    global_search = GA(
        selection=op.sel_tournament,
        crossover=op.cx_uniform,
        mutation=op.mut_gaussian,
        sigma=0.2,
        mut_indpb=0.15,
        cxpb=0.9,
        tournsize=3
    )

    global_search_1 = GA(
        selection=op.sel_tournament,
        crossover=op.cx_uniform,
        mutation=op.mut_gaussian,
        sigma=0.2,
        mut_indpb=0.15,
        cxpb=0.9,
        tournsize=3
    )

    global_search_2 = GA(
        selection=op.sel_tournament,
        crossover=op.cx_uniform,
        mutation=op.mut_gaussian,
        sigma=0.2,
        mut_indpb=0.15,
        cxpb=0.9,
        tournsize=3
    )


    # Create a neighbour of a possibile solution
    def get_neighbour(problem, current_solution):
        neighbour = current_solution.copy()
        index = np.random.randint(0, len(current_solution))
        _min, _max = problem.param_bounds
        neighbour[index] = np.random.uniform(_min, _max)
        return neighbour


    def get_neighbour_1(problem, current_solution):
        neighbour = current_solution.copy()
        index = np.random.randint(0, len(current_solution))
        _min, _max = problem.param_bounds
        neighbour[index] = np.random.uniform(_min, _max)
        return neighbour


    def get_neighbour_2(problem, current_solution):
        neighbour = current_solution.copy()
        index = np.random.randint(0, len(current_solution))
        _min, _max = problem.param_bounds
        neighbour[index] = np.random.uniform(_min, _max)
        return neighbour


    # Define the local search method
    local_search = HC(generate_neighbour=get_neighbour)
    local_search_1 = HC(generate_neighbour=get_neighbour_1)
    local_search_2 = HC(generate_neighbour=get_neighbour_2)

    # Compose the global and local search method for a Memetic Algorithm
    optimizer = MA(global_search=global_search.evolve_population, sel_for_refinement=op.sel_best,
                   local_search=local_search.stochastic_var, frequency=0.1, intensity=10)

    optimizer_1 = MA(global_search=global_search_1.evolve_population, sel_for_refinement=op.sel_best,
                     local_search=local_search_1.stochastic_var, frequency=0.1, intensity=10)

    optimizer_2 = MA(global_search=global_search_2.evolve_population, sel_for_refinement=op.sel_best,
                     local_search=local_search_2.stochastic_var, frequency=0.1, intensity=10)

    res_ae = optimizer.optimize(problem_ae, 2, max_gen=2, verbose=True, seed=42)
    res_oneclass = optimizer_1.optimize(problem_oneclass, 2, max_gen=2, verbose=True, seed=42)
    res_iso_forest = optimizer_2.optimize(problem_iso_forest, 2, max_gen=2, verbose=True, seed=42)
    print(res_oneclass)
    print(res_iso_forest)
    print(res_ae)

    pca_test_reduced = X_test_5perc.copy()
    expectation_values_all_samples_oneclass = []
    expectation_values_all_samples_iso_forest = []
    expectation_values_all_samples_ae = []

    for features in pca_test_reduced:
        reconstruction = calculate_expectation_value(circuit, features, res_oneclass.x)
        expectation_values_all_samples_oneclass.append(reconstruction)

    for features in pca_test_reduced:
        reconstruction = calculate_expectation_value(circuit, features, res_iso_forest.x)
        expectation_values_all_samples_iso_forest.append(reconstruction)

    for features in pca_test_reduced:
        reconstruction = calculate_expectation_value(circuit, features, res_ae.x)
        expectation_values_all_samples_ae.append(reconstruction)

    anomaly_scores_ae = model.predict(np.array(expectation_values_all_samples_ae))
    anomaly_scores_oc = one_class.decision_function(expectation_values_all_samples_oneclass)
    anomaly_scores_if = iso_forest.decision_function(expectation_values_all_samples_iso_forest)
    #anomaly_scores_if = iso_model.decision_function(expectation_values_all_samples_iso_forest)

    error_df = pd.DataFrame({
        'reconstruction_error': anomaly_scores_if,
        'true_class': y_test_5perc
    })

    threshold_oc = np.percentile(anomaly_scores_oc, 95)
    threshold_if = np.percentile(anomaly_scores_if, 95)
    threshold_ae = np.percentile(anomaly_scores_ae, 95)

    # Classifica come anomalie o normali in base al threshold
    y_pred_oc = (anomaly_scores_oc > threshold_oc).astype(int)
    y_pred_if = (anomaly_scores_if > threshold_if).astype(int)
    y_pred_ae = (anomaly_scores_ae > threshold_ae).astype(int)

    # Calcolare l'AUC per ogni modello
    # auc_oc = roc_auc_score(y_test_5perc, y_pred_oc)
    # auc_if = roc_auc_score(y_test_5perc, y_pred_if)
    # auc_ae = roc_auc_score(np.ravel(y_test_5perc), np.mean(y_pred_ae, axis=1))

    # Tracciare le curve ROC
    fpr_oc, tpr_oc, _ = roc_curve(y_test_5perc, anomaly_scores_oc)
    fpr_if, tpr_if, _ = roc_curve(y_test_5perc, anomaly_scores_if)
    fpr_ae, tpr_ae, _ = roc_curve(np.ravel(y_test_5perc), np.mean(anomaly_scores_ae, axis=1))

    # Calcolare l'AUC per ogni modello
    auc_oc = auc(fpr_oc, tpr_oc)
    auc_if = auc(fpr_if, tpr_if)
    auc_ae = auc(fpr_ae, tpr_ae)


    # Visualizzare le curve ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_oc, tpr_oc, color='green', label=f'OC Model AUC = {auc_oc:.2f}')
    plt.plot(fpr_if, tpr_if, color='blue', label=f'IF Model AUC = {auc_if:.2f}')
    plt.plot(fpr_ae, tpr_ae, color='red', label=f'Autoencoder AUC = {auc_ae:.2f}')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')  # Linea diagonale
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig('graphics/roc_curves_compared_f2.pdf', dpi=300, bbox_inches='tight')
    plt.close()