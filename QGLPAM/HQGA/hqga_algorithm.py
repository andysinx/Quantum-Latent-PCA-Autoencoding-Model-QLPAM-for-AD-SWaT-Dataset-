import copy
import math
import random
import time

import numpy as np
from qiskit import transpile
from tqdm import tqdm

from . import hqga_utils


def generate_random_parameter_binds(num_parameters):
    parameter_binds = []

    for _ in range(num_parameters):
        parameter_binds.append(random.uniform(-math.pi, math.pi))

    return parameter_binds


def runQGA(device_features, circuit, params, problem):
    """Function that runs HQGA

    Args:
        device_features (hqga_utils.device): information about the device used to run HQGA
        circuit (qiskit.QuantumCircuit): circuit to be run during HQGA iterations
        params (hqga_utils.Parameters): information about the hyper-parameters of HQGA
        problem (problems.Problem): information about the problem to be solved

    Returns:
        gBest (hqga_utils.globalBest): object that stores the information about the best solution found during the evolution
        chromosome_evolution (list): object that stores the population generated in each iteration
        bests (list): object that stores the best solutions found during each iteration
    """
    chromosome_evolution = []
    bests = []
    # gen=0
    theta = hqga_utils.initializeTheta(circuit, params.epsilon_init)
    gBest = hqga_utils.globalBest()
    list_qubit_gate, list_qubit_entang, list_qubit_mutation, list_qubit_X = hqga_utils.initializeLists(circuit)
    dict_chr = []
    flag_index_best = False
    # while gen<=max_gen:
    l_gen = range(params.max_gen + 1)
    if params.progressBar:
        l_gen = tqdm(range(params.max_gen + 1), desc="Generations")
    for gen in l_gen:
        # print("\n########## generation #########", gen)
        hqga_utils.applyMultiHadamardOnList(circuit, list_qubit_gate)
        hqga_utils.applyMultiRotationOnList(circuit, theta, list_qubit_gate)
        hqga_utils.applyXOnList(circuit, list_qubit_X, dict_chr)
        if gen != 0:
            circuit.barrier()
            hqga_utils.applyEntanglementOnList(circuit, index_best, list_qubit_entang, theta)
            circuit.barrier()
            hqga_utils.applyMutationOnListWithinRange(circuit, params.prob_mut, list_qubit_mutation, theta)
        circuit.barrier()

        hqga_utils.applyMeasureOperator(circuit)
        # Draw the circuit
        if params.draw_circuit:
            print(circuit.draw(output="text", fold=500))
            print("Circuit depth is ", circuit.depth())

        # Execute the circuit on the qasm simulator

        while True:
            circuit.name = str(params.qobj_id) + str(gen)

            try:
                if device_features.real:
                    qc_routed = transpile(
                        circuits=circuit,
                        backend=device_features.device,
                        initial_layout=None,
                        optimization_level=3,
                        layout_method='sabre',
                        routing_method='basic'
                    )
                    job = device_features.device.run(qc_routed)
                else:
                    # feature_to_bind = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
                    feature_to_bind = []
                    if gen == 0:
                        iterations = (circuit.num_parameters - circuit.num_qubits) // 4
                        for i in range(1, iterations + 1):
                            tmp = problem.data[i * 4]
                            feature_to_bind.extend(tmp)
                        rand_params = generate_random_parameter_binds(num_parameters=len(circuit.qubits))
                        bind_params = np.concatenate((feature_to_bind, rand_params))
                    else:
                        feature_to_bind = [c for chrs in chromosome_evolution for chr in chrs for c in chr]
                        if len(feature_to_bind) < circuit.num_parameters:
                            rand_params = generate_random_parameter_binds(num_parameters=(circuit.num_parameters - len(feature_to_bind)))
                            bind_params = np.concatenate((feature_to_bind, rand_params))
                        elif len(feature_to_bind) > circuit.num_parameters:
                            bind_params = feature_to_bind[:circuit.parameters]
                        else:
                            bind_params = feature_to_bind

                    bounded_circuit = circuit.assign_parameters(bind_params)

                    bounded_circuit.measure_all()

                    qc_routed = transpile(
                        circuits=bounded_circuit,
                        backend=device_features.device,
                        initial_layout=None,
                        optimization_level=3,
                        layout_method='sabre',
                        routing_method='basic',
                        coupling_map=device_features.coupling_map,
                        basis_gates=device_features.basis_gates,
                    )

                    job = device_features.device.run(qc_routed, shots=params.num_shots)

                    while not job.done():
                        print("Job is still running...")
                        time.sleep(2)

                # Grab results from the job
                result = job.result()
                break

            except Exception as e:
                print(f"Error in job execution: {e}")

        # Returns counts
        counts = result.get_counts(qc_routed)

        # compute fitness evaluation
        classical_chromosomes = hqga_utils.fromQtoC(hqga_utils.getMaxProbKey(counts))
        classical_chromosomes = [item for item in classical_chromosomes if len(item) > 1]
        if params.verbose:
            print("\nChromosomes", classical_chromosomes)

        l_sup = []
        for c in classical_chromosomes:
            l_sup.append(problem.convert(c))
        chromosome_evolution.append(l_sup)
        if params.verbose:
            print("Phenotypes:", l_sup)

        fitnesses = hqga_utils.computeFitnesses(classical_chromosomes, problem.evaluate)
        if params.verbose:
            print("Fitness values:", fitnesses)
        if gen != 0:
            previous_best = index_best
        best_fitness, index_best = hqga_utils.computeBest(fitnesses, problem.isMaxProblem())
        if params.verbose:
            print("Best fitness", best_fitness, "; index best ", index_best)
        if gen == 0:
            gBest.chr = classical_chromosomes[index_best]
            gBest.phenotype = problem.convert(gBest.chr)
            gBest.fitness = best_fitness
            gBest.gen = params.pop_size
        else:
            flag_index_best = previous_best != index_best
            if hqga_utils.isBetter(best_fitness, gBest.fitness, problem.isMaxProblem()):
                gBest.chr = classical_chromosomes[index_best]
                gBest.phenotype = problem.convert(gBest.chr)
                gBest.fitness = best_fitness
                gBest.gen = (gen + 1) * params.pop_size
        bests.append([problem.convert(gBest.chr), gBest.fitness, gBest.chr])

        list_qubit_gate, list_qubit_entang, list_qubit_mutation, list_qubit_X = hqga_utils.computeLists(circuit,
                                                                                                        index_best,
                                                                                                        4,
                                                                                                        1)

        if params.elitism is not hqga_utils.ELITISM_Q:
            # update
            dict_chr = hqga_utils.create_dict_chr(circuit, classical_chromosomes)

            if params.elitism is hqga_utils.ELITISM_R:
                if flag_index_best:
                    hqga_utils.resetThetaReinforcement(dict_chr, theta, old_theta, previous_best,
                                                       problem.dim * problem.num_bit_code)
                old_theta = copy.deepcopy(theta)
                hqga_utils.updateThetaReinforcementWithinRange(dict_chr, theta, params.epsilon, index_best,
                                                               problem.dim * problem.num_bit_code)
            elif params.elitism is hqga_utils.ELITISM_D:
                hqga_utils.updateListXElitismD(circuit, index_best, list_qubit_gate, list_qubit_X)
            else:
                raise Exception("Value for elitism is not valid.")

        # hqga_utils.resetCircuit(circuit)
        # gen+=1

    gBest.display()
    print("The number of fitness evaluations is: ", params.pop_size * (params.max_gen + 1))
    return gBest, chromosome_evolution, bests
