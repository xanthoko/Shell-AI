# -*- coding: utf-8 -*-
"""Validate_submission.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Qx3SfPKBrVhVrDWoW_tBHCDDvRa5OlgT
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from cost import get_max_supply
from loader import load_infrastructure


def check_constraint_1(_ds: np.ndarray) -> None:
  a = len(_ds[_ds<0])
  assert a == 0, f'Constraint 1 failed. {a} negative values in DS found.'
  print('Constraint 1 ... OK')

def check_constraint_2_3(scs: list[int], fcs: list[int], _parking_slots: list[int]) -> None:
  for i in range(100):
    tps = _parking_slots[i]
    assert scs[i] >= 0, f'Constraint 2 failed for {i}. SCS={scs[i]}'
    assert fcs[i] >= 0, f'Constraint 2 failed for {i}. FCS={fcs[i]}'
    assert scs[i] + fcs[i] <= tps, f'Constraint 3 failed for {i}. TPS={tps} CHARGERS={scs[i]+fcs[i]}'
  print('Constraint 2 ... OK')
  print('Constraint 3 ... OK')

def check_constraint_4(_cscs: list[int], _cfcs: list[int], _pscs: list[int], _pfcs: list[int]) -> None:
  for i in range(100):
    assert _cscs[i] >= _pscs[i], f'Constraint 4 failed for {i}. Cur={_cscs[i]} Prev={_pscs}'
    assert _cfcs[i] >= _pfcs[i], f'Constraint 4 failed for {i}. Cur={_cfcs[i]} Prev={_pfcs}'
  print('Constraint 4 ... OK')

def check_constraint_5(_ds: np.ndarray, _scs: list[int], _fcs: list[int]) -> None:
  sum_of_cols = _ds.sum(axis=0)

  for i in range(100):
    a = sum_of_cols[i]
    b = get_max_supply((_scs[i], _fcs[i]))
    assert a-b <= 10**-2, f'Constraint 5 failed for col {i}. Sum={a} - Smax={b}'
  print('Constraint 5 ... OK')

def check_constraints(_ds: np.ndarray,
                      _scs: list[int],
                      _fcs: list[int],
                      _prev_scs: list[int],
                      _prev_fcs: list[int],
                      _parking_slots: list[int]):
  try:
    check_constraint_1(_ds)
    check_constraint_2_3(_scs, _fcs, _parking_slots)
    check_constraint_4(_scs, _fcs, _prev_scs, _prev_fcs)
    check_constraint_5(_ds, _scs, _fcs)
  except AssertionError as e:
    print('[ERROR]')
    print(e)


def validate(submission_data, parking_slots, prev_scs, prev_fcs) -> tuple[list[int], list[int]]:
    chargers = submission_data[0:200]
    scs_lst = chargers[:100].value.to_list()
    fcs_lst = chargers[100:].value.to_list()
    ds = submission_data[200:]

    demand_supply_19 = ds.value.to_numpy(dtype='float32').reshape((4096, 100))

    check_constraints(demand_supply_19,
                      scs_lst,
                      fcs_lst,
                      prev_scs,
                      prev_fcs,
                      parking_slots)
    return scs_lst, fcs_lst


def main(file_name: str) -> None:
    submission_data = pd.read_csv(file_name)
    half_size = submission_data.shape[0]//2

    infra = load_infrastructure()
    parking_slots = infra.total_parking_slots.to_list()

    print('\n[INFO] Checking 2019')
    scs_19, fcs_19 = validate(submission_data[0:half_size],
                              parking_slots,
                              infra.scs.to_list(),
                              infra.fcs.to_list())
    print('[INFO] 2019 ... OK\n')

    print('\n[INFO] Checking 2020')
    validate(submission_data[half_size:],
             parking_slots,
             scs_19,
             fcs_19)
    print('[INFO] 2020 ... OK')


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print('[ERROR] Please specify a .csv file to validate')
        exit()
    
    main(args[0])
