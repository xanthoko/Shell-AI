from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

output_dir = 'outputs/'
dataset_dir = 'dataset/'


def create_submission_data() -> pd.DataFrame:
    sample_data = pd.read_csv(dataset_dir + 'sample_submission.csv')

    chargers_19 = pd.read_csv(output_dir + 'chargers_2019.csv')
    chargers_20 = pd.read_csv(output_dir + 'chargers_2020.csv')
    ds_19 = np.load(output_dir + 'ds_2019.npy')
    ds_20 = np.load(output_dir + 'ds_2020.npy')

    # Split the sample into 2019 and 2020 sub-dataframes
    half_size = sample_data.shape[0]//2
    half_19 = sample_data[0:half_size]
    half_20 = sample_data[half_size:]

    # Get the 2019 chargers (SCS and FCS) and demand points
    half_19_chargers = half_19[0:200]
    half_19_demand_points = half_19[200:]
    # Get the 2020 chargers (SCS and FCS) and demand points
    half_20_chargers = half_20[0:200]
    half_20_demand_points = half_20[200:]
    
    # We will make our submission by concatenating the 2019 and 2020 submissions
    
    # Overwrite with the calculated 2019 supply points
    half_19_chargers.value = pd.concat([chargers_19.scs,chargers_19.fcs]).astype("float64").tolist()
    # Overwrite with the calculated 2019 DS values
    half_19_demand_points.value = ds_19.reshape(ds_19.shape[0]*ds_19.shape[1],-1).squeeze()
    
    # Overwrite with the calculated 2020 supply points
    half_20_chargers.value = pd.concat([chargers_20.scs,chargers_20.fcs]).astype("float64").tolist()
    # Overwrite with the calculated 2020 DS values
    half_20_demand_points.value = ds_20.reshape(ds_20.shape[0]*ds_20.shape[1],-1).squeeze()

    result = pd.concat([half_19_chargers,half_19_demand_points,half_20_chargers,half_20_demand_points])
    result.to_csv(output_dir + 'submission.csv')


if __name__ == '__main__':
    create_submission_data()
