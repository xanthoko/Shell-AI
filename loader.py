from __future__ import annotations

import csv
import numpy as np
import pandas as pd


dataset_dir = 'dataset/'
output_dir = 'outputs/'


def load_demand_points(year: int) -> pd.DataFrame:
    historic = pd.read_csv(dataset_dir + 'Demand_History.csv')
    prophet_predictions = pd.read_csv(dataset_dir + 'prophet_2019_2020.csv')
    if year == 2019:
        predictions = prophet_predictions.drop(['2020'], axis=1).rename(columns={'2019': 'value'})
    elif year == 2020:
        predictions = prophet_predictions.drop(['2019'], axis=1).rename(columns={'2020': 'value'})

    predictions.insert(1, 'x', historic.x_coordinate)
    predictions.insert(2, 'y', historic.y_coordinate)

    return predictions


def load_infrastructure() -> pd.DataFrame:
    existing = pd.read_csv(dataset_dir + 'exisiting_EV_infrastructure_2018.csv')
    return existing.rename({
        'x_coordinate': 'x',
        'y_coordinate': 'y',
        'existing_num_SCS': 'scs',
        'existing_num_FCS': 'fcs'
    }, axis=1)


def load_distances() -> tuple[np.ndarray, np.ndarray]:
    return np.load(dataset_dir + 'distance.npy'), np.load(dataset_dir + 'reverse_proximity.npy')


def load_previous_chargers(year: int) -> dict[int, tuple[int, int]]:
    if year == 2019:
        existing_infra = load_infrastructure()
    elif year == 2020:
        existing_infra = pd.read_csv(output_dir + 'chargers_2019.csv')
    else:
        raise ValueError('Invalid year')
    return {ind: (int(x.scs), int(x.fcs)) for ind, x in existing_infra.iterrows()}


def output_chargers(supply_chargers: dict[int, tuple[int, int]], year: int) -> None:
    with open(output_dir + f'chargers_{year}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scs', 'fcs'])
        for _, (scs, fcs) in supply_chargers.items():
            writer.writerow([scs, fcs])


def output_distribution(ds: np.ndarray, year: int) -> None:
    np.save(output_dir + f'ds_{year}.npy', ds)
