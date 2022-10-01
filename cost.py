from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from boost import get_redundant_chargers
from definitions import Genome
from definitions import SLOW_CHARGE_CAP
from definitions import FAST_CHARGE_CAP


def get_max_supply(supply_charge: tuple[int, int]) -> int:
    return SLOW_CHARGE_CAP * supply_charge[0] + FAST_CHARGE_CAP * supply_charge[1]


def distribute_supply(supply_charges: Genome,
                      sorted_demands: list[tuple[int, float]],
                      reverse_proximity: np.ndarray) -> np.ndarray:
    ds = np.zeros((4096, 100))
    dd = sorted_demands.copy()
    levels = [get_max_supply(supply_charges[i]) for i in range(100)]

    for i in range(4096):
        sp_index = 0

        while sp_index < 100:
            demand_index, value = dd[i]
            selected_supply = reverse_proximity[demand_index][sp_index]
            supply_level = levels[selected_supply]

            if supply_level > 0:
                given_supply = min(supply_level, value)
                ds[demand_index, selected_supply] = given_supply
                levels[selected_supply] -= given_supply
                dd[i] = (demand_index, value - given_supply)
            
            if dd[i][1] == 0:
                break
            sp_index += 1
    return ds


def check_constraint_1(_ds: np.ndarray) -> None:
    a = len(_ds[_ds < 0])
    assert a == 0, f'Constraint 1 failed. {a} negative values in DS found.'


def check_constraint_2_3(_supply_charges: Genome,
                         parking_slots: list[int]) -> None:
    for i in range(100):
        scs = _supply_charges[i][0]
        fcs = _supply_charges[i][1]
        tps = parking_slots[i]
        assert scs >= 0, f'Constraint 2 failed for {i}. SCS={scs}'
        assert fcs >= 0, f'Constraint 2 failed for {i}. FCS={fcs}'
        assert scs + fcs <= tps, f'Constraint 3 failed for {i}. TPS={tps}'


def check_constraint_4(_supply_charges: Genome,
                       previous_charges: Genome) -> None:
    for i in range(100):
        cs, cf = _supply_charges[i]
        ps, pf = previous_charges[i]
        assert cs >= ps, f'Constraint 4 failed for {i}. cs={cs} ps={ps}'
        assert cf >= pf, f'Constraint 4 failed for {i}. cf={cf} pf={pf}'


def check_constraint_5(_ds: np.ndarray,
                       _supply_charges: Genome) -> None:
    sum_of_cols = _ds.sum(axis=0)

    for i in range(100):
        a = sum_of_cols[i]
        b = get_max_supply(_supply_charges[i])
        assert round(a,
                     3) <= b, f'Constraint 5 failed for col {i}. Sum={a} - Smax={b}'


def check_constraint_6(_ds: np.ndarray, demand_values: list[float]) -> None:
    sum_of_rows = _ds.sum(axis=1)

    for i in range(4096):
        a = sum_of_rows[i]
        b = demand_values[i]
        assert abs(
            a -
            b) < 10**-2, f'Constraint 6 failed for row {i}. Sum={a} - Demand={b}'


def check_constraints(_ds: np.ndarray, _supply_charges: Genome,
                      parking_slots: list[int], previous_charges: dict[int,
                                                                       tuple[int,
                                                                             int]],
                      demand_values: list[float]) -> None:
    check_constraint_1(_ds)
    check_constraint_2_3(_supply_charges, parking_slots)
    check_constraint_4(_supply_charges, previous_charges)
    check_constraint_5(_ds, _supply_charges)
    check_constraint_6(_ds, demand_values)


def get_cost_1(_ds: np.ndarray, distance_matrix: np.ndarray) -> float:
    return np.sum(distance_matrix * _ds, axis=(0, 1))


def get_cost_3(_supply_charges: Genome) -> float:
    cost_3 = 0
    for i in range(100):
        cost_3 += _supply_charges[i][0] + 1.5 * _supply_charges[i][1]

    return cost_3


def get_overall_cost(_ds: np.ndarray, _supply_charges: Genome,
                     distance_matrix: np.ndarray) -> float:
    a, c = 1, 600
    cost_1 = get_cost_1(_ds, distance_matrix)
    cost_3 = get_cost_3(_supply_charges)
    return a * cost_1 + c * cost_3


class Fitness:

    def __init__(
        self,
        sorted_demands: pd.DataFrame,
        reverse_proximity: np.ndarray,
        parking_slots: list[int],
        previous_charges: Genome,
        demand_values: list[float],
        distance_matrix: np.ndarray,
    ) -> None:
        self.sorted_demands = sorted_demands
        self.reverse_proximity = reverse_proximity
        self.parking_slots = parking_slots
        self.previous_charges = previous_charges
        self.demand_values = demand_values
        self.distance_matrix = distance_matrix

    def fitness_function(self, supply_charges: Genome) -> float:
        try:
            ds = distribute_supply(supply_charges, self.sorted_demands,
                                   self.reverse_proximity)
            check_constraints(ds, supply_charges, self.parking_slots,
                              self.previous_charges, self.demand_values)
            cost = get_overall_cost(ds, supply_charges, self.distance_matrix)
            return cost
        except AssertionError as e:
            return sys.maxsize
