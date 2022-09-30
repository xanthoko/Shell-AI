from __future__ import annotations

import numpy as np
import pandas as pd

from cost import check_constraints
from cost import get_max_supply
from loader import load_demand_points
from loader import load_rev_proximity
from loader import load_chargers
from loader import load_demand_points
from loader import load_infrastructure, load_previous_chargers


reverse_proximity = load_rev_proximity()
chargers: pd.DataFrame = load_chargers(2019)
demand_points: pd.DataFrame = load_demand_points(2019)
existing_infra: pd.DataFrame = load_infrastructure()
previous_charges = load_previous_chargers(2019)

demand_values = demand_points.value.to_list()
sorted_demand_points = [
    (int(dp.demand_point_index), dp.value)
    for _, dp in demand_points.sort_values('value', ascending=False).iterrows()
]
parking_slots: list[int] = existing_infra.total_parking_slots.to_list()

sp = {i: (int(x.scs), int(x.fcs)) for i, x in chargers.iterrows()}


# DS
ds = np.zeros((4096, 100))
dd = sorted_demand_points.copy()
levels = [get_max_supply(sp[i]) for i in range(100)]

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

try:
    check_constraints(ds, sp, parking_slots, previous_charges, demand_values)
except AssertionError as e:
    print(e)
