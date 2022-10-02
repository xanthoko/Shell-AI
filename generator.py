# -*- coding: utf-8 -*-
"""
Steps in a Genetic Algorithm
* Initialize population
* Select parents by evaluating their fitness
* Crossover parents to reproduce
* Mutate the offsprings
* Evaluate the offsprings
* Merge offsprings with the main population and sort
"""
from __future__ import annotations

import random
import pickle
import time
from xmlrpc.client import boolean
import numpy as np
import pandas as pd
from numpy.random import choice

from cost import fitness_function
from cost import distribute_supply
from loader import load_distances
from loader import output_chargers
from loader import load_demand_points
from loader import output_distribution
from loader import load_infrastructure
from loader import load_previous_chargers

YEAR = 2019
# POPULATION_SIZE = 350  # Number of individuals in each generation
# GENERATIONS = 200
output_dir = 'outputs/'

demand_points: pd.DataFrame = load_demand_points(YEAR)
existing_infra: pd.DataFrame = load_infrastructure()
distance_matrix, reverse_proximity = load_distances()
previous_charges = load_previous_chargers(YEAR)
parking_slots: list[int] = existing_infra.total_parking_slots.to_list()

demand_values = demand_points.value.to_list()
sorted_demand_points = [
    (int(dp.demand_point_index), dp.value)
    for _, dp in demand_points.sort_values('value', ascending=False).iterrows()
]


def offspring_generator(prob: float) -> dict[int, tuple[int, int]]:
    temp_dict = {}
    for i in range(100):
        if random.random() > prob:
            temp_dict[i] = previous_charges[i]
            continue

        available_parking_slots = parking_slots[i] - previous_charges[i][0] - previous_charges[i][1]
        random_scs = random.randint(0, available_parking_slots)
        random_fcs = random.randint(0, available_parking_slots - random_scs)
        temp_dict[i] = (previous_charges[i][0] + random_scs,
                        previous_charges[i][1] + random_fcs)
    return temp_dict


def crossover(parent1: dict, parent2: dict, start: int,
              end: int) -> tuple[dict, dict]:
    offspring1 = {}
    offspring2 = {}
    # one-point separation
    lim = random.randint(start, end)
    # Generate 1st offsrping
    offspring1.update(dict(list(parent1.items())[:lim]).items())
    offspring1.update(dict(list(parent2.items())[lim:]).items())
    # Generate 2nd offsrping
    offspring2.update(dict(list(parent2.items())[:lim]).items())
    offspring2.update(dict(list(parent1.items())[lim:]).items())
    return (offspring1, offspring2)


def crossover_twopoints(parent1: dict, parent2: dict, start: int = None,
                        end: int = None) -> tuple[dict, dict]:
    offspring1 = {}
    offspring2 = {}
    
    # two-point separation
    point_one = random.randint(2, 80)
    point_two = random.randint(point_one + 5, 95)

    # Generate 1st offsrping
    offspring1.update(dict(list(parent1.items())[:point_one]).items())
    offspring1.update(dict(list(parent2.items())[point_one:point_two]).items())
    offspring1.update(dict(list(parent1.items())[point_two:]).items())
    # Generate 2nd offsrping
    offspring2.update(dict(list(parent2.items())[:point_one]).items())
    offspring2.update(dict(list(parent1.items())[point_one:point_two]).items())
    offspring2.update(dict(list(parent2.items())[point_two:]).items())
    return (offspring1, offspring2)


def random_crossover(parent1: dict, parent2: dict):
    offspring1 = {}
    offspring2 = {}
    for gp1, gp2 in zip(parent1.items(), parent2.items()):
        # random probability
        prob = random.random()
        # if prob is less than 0.45, insert gene from parent 1
        if prob < 0.5:
            offspring1.update({gp1[0]: gp1[1]})
            offspring2.update({gp2[0]: gp2[1]})

        # if prob is higher than 0.5, insert gene from parent 2
        elif prob >= 0.5:
            offspring1.update({gp2[0]: gp2[1]})
            offspring2.update({gp1[0]: gp1[1]})

    return (offspring1, offspring2)

def random_crossover4(parent1: dict, parent2: dict, parent3: dict, parent4: dict):
    offspring1 = {}
    offspring2 = {}
    for gp1, gp2, gp3, gp4 in zip(parent1.items(), parent2.items(), parent3.items(), parent4.items()):
        # random probability
        prob = random.random()
        # if prob is less than 0.45, insert gene from parent 1
        if prob < 0.25:
            offspring1.update({gp1[0]: gp1[1]})
            offspring2.update({gp2[0]: gp2[1]})
        # if prob is higher than 0.5, insert gene from parent 2
        elif prob < 0.5:
            offspring1.update({gp2[0]: gp2[1]})
            offspring2.update({gp1[0]: gp1[1]})
        # if prob is higher than 0.5, insert gene from parent 2
        elif prob < 0.75:
            offspring1.update({gp3[0]: gp3[1]})
            offspring2.update({gp4[0]: gp4[1]})
        # if prob is higher than 0.5, insert gene from parent 2
        elif prob <= 1:
            offspring1.update({gp4[0]: gp4[1]})
            offspring2.update({gp3[0]: gp3[1]})

    return (offspring1, offspring2)


def mutate(offspring: dict, num_of_charges: int) -> dict:
    for _ in range(num_of_charges):
        idx = random.choice(list(offspring.keys()))
        available_parking_slots = parking_slots[idx] - previous_charges[idx][0] - previous_charges[idx][1]

        random_scs = random.randint(0, available_parking_slots)
        random_fcs = random.randint(0, available_parking_slots - random_scs)
        offspring[idx] = (previous_charges[idx][0] + random_scs,
                          previous_charges[idx][1] + random_fcs)
    return offspring


def save_file(solution):
    a_file = open("solution.pkl", "wb")
    pickle.dump(solution, a_file)
    a_file.close()


def load_pckl():
    a_file = open("solution.pkl", "rb")
    output = pickle.load(a_file)
    a_file.close()
    print(output)

##################################################################
POPULATION_SIZE = 1000  # Number of individuals in each generation
GENERATIONS = 500
##################################################################
population = []
for _ in range(POPULATION_SIZE):
    gnome = offspring_generator(random.uniform(0.95, 1))
    score = fitness_function(gnome, sorted_demand_points, reverse_proximity,
                             parking_slots, previous_charges, demand_values,
                             distance_matrix)
    population.append((gnome, score))

# sort the population in increasing order of fitness score
population = sorted(population, key=lambda x: x[1])

# current generation
generation = 1
best_per_population = []

dynamic_pm = False
dynamic_pc = False
roullete = False
pc = 0.8 #Probability of crossover 
pm = 0.25 #Probability of mutation

while generation <= GENERATIONS:
    # Perform Elitism, that mean 10% of fittest population goes to the next generation
    elit = 10
    s = (15 * POPULATION_SIZE) // 100
    # s = elit
    new_generation = population[:s]

    # From 50% of fittest population, Individuals will mate to produce offspring
    s = (85 * POPULATION_SIZE) // 100
    # s = POPULATION_SIZE - elit
    # s = s//2

    if roullete:
        # Computes the totallity of the population fitness
        population_fitness = sum([chromosome[1] for chromosome in population])

        # Computes for each chromosome the probability 
        chromosome_probabilities = [chromosome[1]/population_fitness for chromosome in population]

        # Selects one chromosome based on the computed probabilities
        population_1 = [chromosome[0] for chromosome in population]
    
    # Dynamic calculation of pc & pm
    if dynamic_pm:
        pm = generation/GENERATIONS
    if dynamic_pc:
        pc = 1 - generation/GENERATIONS
    limit = POPULATION_SIZE//4    
    
    # for _ in range(s):
    while len(new_generation) < POPULATION_SIZE:
        
        if roullete:
        # Selection οf chromosomes based on the computed probabilities
            parent1 = choice(population_1, p=chromosome_probabilities) #random.choice(population[:50])
            parent2 = choice(population_1, p=chromosome_probabilities) #random.choice(population[:50])
        else:
            limit = POPULATION_SIZE//5

            parent1 = random.choice(population[:limit])[0]
            parent2 = random.choice(population[:limit])[0] #random.choice(population[:50])

        # Crossover
        if random.uniform(0, 1) <= pc:
            match random.randint(1,4):
                case 1:
                    # child1, child2 = crossover(parent1, parent2, 2, 98)
                    child1 = crossover(parent1, parent2, 2, 98)[0]
                    # child2 = crossover(parent1, parent2, 2, 98)[0]
                case 2:
                    # child1, child2 = random_crossover(parent1, parent2)
                    child1 = random_crossover(parent1, parent2)[0]
                    # child2 = random_crossover(parent1, parent2)[0]
                case 3:
                    # child1, child2 = crossover_twopoints(parent1, parent2)
                    child1 = crossover_twopoints(parent1, parent2)[0]
                    # child2 = crossover_twopoints(parent1, parent2)[0]
                case 4:
                    if roullete:
                        parent3 = choice(population_1, p=chromosome_probabilities)
                        parent4 = choice(population_1, p=chromosome_probabilities)
                    else:
                        parent3 = random.choice(population[:limit])[0]
                        parent4 = random.choice(population[:limit])[0]
                    
                    child1, child2 = random_crossover4(parent1, parent2, parent3, parent4)
                case _:
                    child1, child2 = crossover(parent1, parent2, 2, 98)
        else:
            child1, child2 = parent1, parent2
        
        # Mutate
        if random.uniform(0, 1) < pm:
            s = 5
            e = 20
            child1 = mutate(child1, random.randint(s, e))
            # child2 = mutate(child2, random.randint(s, e))

        # Fitness calculation
        new_generation.append(
            (child1,
             fitness_function(child1, sorted_demand_points, reverse_proximity,
                              parking_slots, previous_charges, demand_values,
                              distance_matrix)))
        # new_generation.append(
        #     (child2,
        #      fitness_function(child2, sorted_demand_points, reverse_proximity,
        #                       parking_slots, previous_charges, demand_values,
        #                       distance_matrix)))
    population = new_generation

    # sort the population in increasing order of fitness score
    population = sorted(new_generation, key=lambda x: x[1])

    best_per_population.append((population[0][0], population[0][1], generation))
    if roullete:
        f = len(set([str(z[0]) for z in population]))
    else:

        f = len(set([str(z[0]) for z in population[:limit]]))
    if generation % 10 == 0 or generation == 1:
        print(f'Gen.: {generation}\t\t Cost: {population[0][1]}\t Worst cost: {population[-1][1]}\t Unique values: {f}')
    # d = [c[1] for c in population]
    # print(f'Unique chromosomes in generation {generation}: {len(set(d))}')

    generation += 1

print('###########################################')
print(f'Year:{YEAR}\nGenerations: {GENERATIONS}\
        \nPop_size: {POPULATION_SIZE}\ndynamic_pc:{dynamic_pc}\
        \ndynamic_pm:{dynamic_pm}\npc:{pc}\npm:{pm}\nroullete:{roullete}\n')
best_population, best_cost = population[0]
print(f'Gen.: {generation-1}\t\t Cost: {best_cost}')
best_ds = distribute_supply(best_population, sorted_demand_points,
                            reverse_proximity)

output_chargers(best_population, YEAR)
output_distribution(best_ds, YEAR)

best_per_population = sorted(best_per_population, key=lambda x: x[1])
# pop_size + num_of_generations + pc + mc
file_name = f'{output_dir}exports/{POPULATION_SIZE}_{GENERATIONS}_{pc}_{pm}_{time.strftime("%Y%m%d-%H%M%S")}_best_per_population'

a_file = open(file_name, "wb")
pickle.dump(best_per_population, a_file)
a_file.close()
print(f'In best_per_population list, Best Cost: {best_per_population[0][1]}\t Worst cost: {best_per_population[-1][1]}')
dd = [c[1] for c in best_per_population]
print(f'Unique chromosomes in best_per_population: {len(set(dd))}')
