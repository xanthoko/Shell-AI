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
import argparse
import pandas as pd

from cost import Fitness
from cost import distribute_supply
from loader import load_distances
from loader import output_chargers
from loader import load_demand_points
from loader import output_distribution
from loader import load_infrastructure
from loader import load_previous_chargers


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


def crossover_twopoints(parent1: dict,
                        parent2: dict,
                        start: int = None,
                        end: int = None) -> tuple[dict, dict]:
    offspring1 = {}
    offspring2 = {}

    # two-point separation
    # point_one = random.randint(start, start + 45)
    # point_two = random.randint(end, end + 45)

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


def run_evolution():
    population = []
    for _ in range(POPULATION_SIZE):
        gnome = offspring_generator(random.uniform(0.95, 1))
        score = fit.fitness_function(gnome)
        population.append((gnome, score))

    # sort the population in increasing order of fitness score
    population = sorted(population, key=lambda x: x[1])

    # current generation
    generation = 1
    best_per_population = []

    pc = 0.8  # Probability of crossover
    pm = 0.25  # Probability of mutation

    while generation <= GENERATIONS:
        # Perform Elitism, that mean 10% of fittest population
        # goes to the next generation
        s = (10 * POPULATION_SIZE) // 100
        new_generation = population[:s]

        # From 50% of fittest population, Individuals will mate to produce offspring
        s = (90 * POPULATION_SIZE) // 100
        s = s // 2

        # Computes the totallity of the population fitness
        population_fitness = sum([chromosome[1] for chromosome in population])

        # Computes for each chromosome the probability
        chromosome_probabilities = [
            chromosome[1] / population_fitness for chromosome in population
        ]

        # Selects one chromosome based on the computed probabilities
        # choice(population, p=chromosome_probabilities)
        population_1 = [chromosome[0] for chromosome in population]

        # Dynamic calculation of pc & pm
        pm = generation / GENERATIONS
        pc = 1 - pm

        for _ in range(s):
            # Selection οf chromosomes based on the computed probabilities
            # parent1 = choice(population_1, p=chromosome_probabilities) #random.choice(population[:50])
            # parent2 = choice(population_1, p=chromosome_probabilities) #random.choice(population[:50])
            parent1 = random.choice(population[:POPULATION_SIZE // 2])[0]
            parent2 = random.choice(
                population[:POPULATION_SIZE //
                           2])[0]  #random.choice(population[:50])

            # Crossover
            if random.uniform(0, 1) < pc:
                rr = random.randint(1, 3)
                if rr == 1:
                    child1, child2 = crossover(parent1, parent2, 2, 98)
                elif rr == 2:
                    child1, child2 = crossover_twopoints(parent1, parent2)
                elif rr == 3:
                    child1, child2 = random_crossover(parent1, parent2)
                else:
                    child1, child2 = crossover(parent1, parent2, 40, 60)
            else:
                child1, child2 = parent1, parent2

            # child1, child2 = crossover(parent1[0], parent2[0], 40, 60)

            # Mutate
            if random.uniform(0, 1) < pm:
                child1 = mutate(child1, random.randint(10, 20))
                child2 = mutate(child2, random.randint(10, 20))

            # Fitness calculation
            new_generation.append((child1, fit.fitness_function(child1)))
            new_generation.append((child2, fit.fitness_function(child2)))

        population = new_generation

        # sort the population in increasing order of fitness score
        population = sorted(new_generation, key=lambda x: x[1])

        best_per_population.append((population[0][0], population[0][1], generation))
        print(
            f'Gen.: {generation}\t\t Cost: {population[0][1]}\t Worst cost: {population[-1][1]}'
        )

        generation += 1

    print('######################################')
    best_population, best_cost = population[0]
    print(f'Gen.: {generation-1}\t\t Cost: {best_cost}')
    best_ds = distribute_supply(best_population, sorted_demand_points,
                                reverse_proximity)

    output_chargers(best_population, YEAR)
    output_distribution(best_ds, YEAR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Genetic Algorithm')
    parser.add_argument('year',
                        metavar='YEAR',
                        type=int,
                        help='the year to predict on',
                        choices={2019, 2020})
    parser.add_argument('-g',
                        '--generations',
                        metavar='\b',
                        type=int,
                        default=100,
                        help='number of generations to run')
    parser.add_argument('-p',
                        '--population',
                        metavar='\b',
                        type=int,
                        default=200,
                        help='size of population')

    args = parser.parse_args()
    YEAR = args.year
    GENERATIONS = args.generations
    POPULATION_SIZE = args.population
    print(YEAR, GENERATIONS, POPULATION_SIZE)
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

    fit = Fitness(sorted_demand_points, reverse_proximity, parking_slots,
                  previous_charges, demand_values, distance_matrix)
