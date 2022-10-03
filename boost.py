from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from definitions import Genome
from definitions import SLOW_CHARGE_CAP
from definitions import FAST_CHARGE_CAP
from loader import load_ds
from loader import load_chargers
from loader import output_chargers
from loader import load_previous_chargers


def get_redundant_chargers(chargers: tuple[int, int], supply_used: float,
                           previous_chargers: tuple[int, int]) -> tuple[int, int]:
    scs, fcs = chargers
    available_supply = SLOW_CHARGE_CAP * scs + FAST_CHARGE_CAP * fcs
    diff = available_supply - supply_used

    # the diff has to be greater than 200 to be able to remove at least one charger
    if diff < 200:
        return 0, 0

    # check how many fcs can be removed
    removable_fcs = fcs - previous_chargers[1]
    unused_fcs = diff // 400
    removed_fcs = min(removable_fcs, unused_fcs)

    diff -= removed_fcs * 400

    # check how many scs can be removed
    removable_scs = scs - previous_chargers[0]
    unused_scs = diff // 200
    removed_scs = min(removable_scs, unused_scs)

    return removed_scs, removed_fcs


def remove_excess_supply(chargers: Genome, previous_chargers: Genome,
                         ds: np.ndarray) -> None:
    total_removed_scs = 0
    total_removed_fcs = 0
    supplies_used = np.sum(ds, axis=0)

    for i, charger in chargers.items():
        removed_scs, removed_fcs = get_redundant_chargers(charger, supplies_used[i],
                                                          previous_chargers[i])

        total_removed_scs += removed_scs
        total_removed_fcs += removed_fcs

        # update genome
        chargers[i] = (charger[0] - removed_scs, charger[1] - removed_fcs)

    saved_cost = (total_removed_scs + 1.5 * total_removed_fcs) * 600
    print('[INFO] - Excess supply removed')
    print(f'{total_removed_scs} scs removed')
    print(f'{total_removed_fcs} fcs removed')
    print(f'{saved_cost} cost saved')


def convert_scs_to_fcs(chargers: Genome, previous_chargers: Genome) -> None:
    converted_scs = 0
    for i, (scs, fcs) in chargers.items():
        available_scs = int(scs) - previous_chargers[i][0]
        convertable_scs = available_scs // 2 * 2
        added_fcs = convertable_scs // 2
        chargers[i] = (scs - convertable_scs, fcs + added_fcs)

        converted_scs += convertable_scs

    saved_cost = (converted_scs // 2 * 0.5) * 600
    print('[INFO] - SCS converted to FCS')
    print(f'{converted_scs} scs converted')
    print(f'{saved_cost} cost saved')


def main(year: int) -> None:
    ds: np.ndarray = load_ds(year)
    chargers_df: pd.DataFrame = load_chargers(year)
    previous_chargers: Genome = load_previous_chargers(year)

    # covert DataFrame to Genome
    chargers = {}
    for i, sp in chargers_df.iterrows():
        chargers[i] = (sp.scs, sp.fcs)

    remove_excess_supply(chargers, previous_chargers, ds)
    convert_scs_to_fcs(chargers, previous_chargers)

    output_chargers(chargers, year)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('year',
                        metavar='YEAR',
                        type=int,
                        help='The year of the prediction',
                        choices={2019, 2020})
    args = parser.parse_args()
    main(args.year)
