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
import argparse
import pandas as pd

from definitions import Genome
from definitions import Population
from cost import Fitness
from cost import distribute_supply
from loader import load_distance
from loader import output_chargers
from loader import load_rev_proximity
from loader import load_demand_points
from loader import output_distribution
from loader import load_infrastructure
from loader import load_previous_chargers
from boost import convert_scs_to_fcs
from boost import remove_excess_supply


def generate_population(size: int) -> Population:
    population = []
    for _ in range(size):
        gnome = offspring_generator(random.uniform(0.95, 1))
        score = fit.fitness_function(gnome)
        population.append((gnome, score))
    return population


def offspring_generator(prob: float) -> Genome:
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


def crossover(parent1: Genome, parent2: Genome, start: int,
              end: int) -> tuple[Genome, Genome]:
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


def crossover_twopoints(parent1: Genome,
                        parent2: Genome,
                        start: int = None,
                        end: int = None) -> tuple[Genome, Genome]:
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


def random_crossover(parent1: Genome, parent2: Genome) -> tuple[Genome, Genome]:
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


def mutate(offspring: Genome, num_of_charges: int) -> Genome:
    for _ in range(num_of_charges):
        idx = random.choice(list(offspring.keys()))
        available_parking_slots = parking_slots[idx] - previous_charges[idx][
            0] - previous_charges[idx][1]

        random_scs = random.randint(0, available_parking_slots)
        random_fcs = random.randint(0, available_parking_slots - random_scs)
        offspring[idx] = (previous_charges[idx][0] + random_scs,
                          previous_charges[idx][1] + random_fcs)
    return offspring


def run_evolution():
    population = generate_population(POPULATION_SIZE)
    # sort the population in increasing order of fitness score
    population = sorted(population, key=lambda x: x[1])

    for generation in range(GENERATIONS):
        
        pm = generation / GENERATIONS
        pc = 1 - pm
        
        # Perform Elitism, that mean 10% of fittest population
        # goes to the next generation
        s1 = (10 * POPULATION_SIZE) // 100
        new_generation = population[:s1]

        # From 50% of fittest population, Individuals will mate to produce offspring
        s2 = (90 * POPULATION_SIZE) // 100
        s2 = s2 // 2

        for _ in range(s2):
            # Selection οf chromosomes based on the computed probabilities
            parent1 = random.choice(population[:POPULATION_SIZE // 2])[0]
            parent2 = random.choice(population[:POPULATION_SIZE // 2])[0]

            # Crossover
            if random.random() < pc:
                rr = random.randint(1, 3)
                if rr == 1:
                    child1, child2 = crossover(parent1, parent2, 2, 98)
                elif rr == 2:
                    child1, child2 = crossover_twopoints(parent1, parent2)
                elif rr == 3:
                    child1, child2 = random_crossover(parent1, parent2)
                else:
                    child1, child2 = crossover(parent1, parent2, 40, 60)

            if random.random() < pm:
                child1 = mutate(child1, random.randint(10, 20))
                child2 = mutate(child2, random.randint(10, 20))

            # Fitness calculation
            new_generation.append((child1, fit.fitness_function(child1)))
            new_generation.append((child2, fit.fitness_function(child2)))

        # sort the population in increasing order of fitness score
        population = sorted(new_generation, key=lambda x: x[1])
        best_gen_genome, best_gen_score = population[0]

        # see unique genomes
        unique_genomes = len(set([str(x[0].values()) for x in population]))
        uniques_scores = len(set([x[1] for x in population]))

        print(f'Gen.: {generation}\t\t Cost: {best_gen_score} \t Uniques: {unique_genomes}')

    print('######################################')
    best_population, best_cost = population[0]
    print(f'Gen.: {generation+1}\t\t Cost: {best_cost}')
    best_ds = distribute_supply(best_population, sorted_demand_points,
                                reverse_proximity)

    # boost the best population
    remove_excess_supply(best_population, previous_charges, best_ds)
    convert_scs_to_fcs(best_population, previous_charges)

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

    output_dir = 'outputs/'

    demand_points: pd.DataFrame = load_demand_points(YEAR)
    existing_infra: pd.DataFrame = load_infrastructure()
    distance_matrix = load_distance()
    reverse_proximity = load_rev_proximity()
    previous_charges: Genome = load_previous_chargers(YEAR)
    parking_slots: list[int] = existing_infra.total_parking_slots.to_list()

    demand_values = demand_points.value.to_list()
    sorted_demand_points = [
        (int(dp.demand_point_index), dp.value)
        for _, dp in demand_points.sort_values('value', ascending=False).iterrows()
    ]

    fit = Fitness(sorted_demand_points, reverse_proximity, parking_slots,
                  previous_charges, demand_values, distance_matrix)

    run_evolution()
