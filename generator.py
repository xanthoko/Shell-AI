# -*- coding: utf-8 -*-
"""Shell_AI_Genetic_Algo.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1E1YPrXqygXmo7i5ChRbvIWxh5q2BE9_f

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
import pandas as pd

from cost import fitness_function
from cost import distribute_supply
from loader import load_distances
from loader import output_chargers
from loader import load_demand_points
from loader import output_distribution
from loader import load_infrastructure
from loader import load_previous_chargers

YEAR = 2020
GENERATIONS = 100
POPULATION_SIZE = 200  # Number of individuals in each generation
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

        available_parking_slots = parking_slots[i] - previous_charges[i][
            0] - previous_charges[i][1]
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


def crossover_twopoints(parent1: dict, parent2: dict, start: int,
                        end: int) -> tuple[dict, dict]:
    offspring1 = {}
    offspring2 = {}
    # two-point separation
    point_one = random.randint(start, start + 15)
    point_two = random.randint(end, end + 15)
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
        print(prob)
        # if prob is less than 0.45, insert gene from parent 1
        if prob < 0.5:
            offspring1.update({gp1[0]: gp1[1]})
            offspring2.update({gp2[0]: gp2[1]})

        # if prob is between 0.45 and 0.90, insert gene from parent 2
        elif prob < 1:
            offspring1.update({gp2[0]: gp2[1]})
            offspring2.update({gp1[0]: gp1[1]})

        # otherwise invert number of chargers between fcs & scf, for maintaining diversity
        # else:
        #   offspring1.update({gp1[0]:(gp1[1][1], gp1[1][0])})
        #   offspring2.update({gp2[0]:(gp2[1][1], gp2[1][0])})

    return (offspring1, offspring2)


def mutate(offspring: dict, num_of_charges: int) -> dict:
    for _ in range(num_of_charges):
        idx = random.choice(list(offspring.keys()))
        available_parking_slots = parking_slots[idx] - previous_charges[idx][
            0] - previous_charges[idx][1]

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


population = []
for _ in range(POPULATION_SIZE):
    gnome = offspring_generator(random.uniform(0.9, 1))
    score = fitness_function(gnome, sorted_demand_points, reverse_proximity,
                             parking_slots, previous_charges, demand_values,
                             distance_matrix)
    population.append((gnome, score))

# sort the population in increasing order of fitness score
population = sorted(population, key=lambda x: x[1])

# current generation
generation = 1
found = False
best_per_population = {}

while generation <= GENERATIONS:
    # Perform Elitism, that mean 10% of fittest population
    # goes to the next generation
    s = (10 * POPULATION_SIZE) // 100
    new_generation = population[:s]

    # From 50% of fittest population, Individuals will mate to produce offspring
    s = (90 * POPULATION_SIZE) // 100
    for _ in range(s):
        # Crossover
        parent1 = random.choice(population[:50])
        parent2 = random.choice(population[:50])
        child1, child2 = crossover(parent1[0], parent2[0], 40, 60)
        # Mutate
        child1 = mutate(child1, random.randint(2, 10))
        child2 = mutate(child2, random.randint(2, 10))

        # Fitness calculation
        new_generation.append(
            (child1,
             fitness_function(child1, sorted_demand_points, reverse_proximity,
                              parking_slots, previous_charges, demand_values,
                              distance_matrix)))
        new_generation.append(
            (child2,
             fitness_function(child2, sorted_demand_points, reverse_proximity,
                              parking_slots, previous_charges, demand_values,
                              distance_matrix)))

    # sort the population in increasing order of fitness score
    population = sorted(new_generation, key=lambda x: x[1])

    best_per_population[generation] = population[0][1]
    print(f'Gen.: {generation}\t\t Cost: {population[0][1]}')

    generation += 1

print('######################################')
best_population, best_cost = population[0]
print(f'Gen.: {generation-1}\t\t Cost: {best_cost}')
best_ds = distribute_supply(best_population, sorted_demand_points,
                            reverse_proximity)

output_chargers(best_population, YEAR)
output_distribution(best_ds, YEAR)