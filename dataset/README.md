# Data files used

## Prediction files
Given the demand values for each demand point for the years 2010 to 2018, we had to predict the demand values for the years 2019 and 2020.

In order to achieve that we used different algorithms and approaches to minimize the MAE. @tolism

- prophet_2019_2020.csv: ...
- ensemble_2019_2020.csv: ...
- v2_ensemble_2019_2020.csv: ...

## Distance based files
Since both the supply and the demand points have fixed coordinates that do not change per year, the distance and the proximity between them remains the same. 

Those values have been calculated, stored to numpy arrays and saved to the following .npy files:

- **distance.npy**: 4096x100 numpy array. The cell (i,j) has the distance of the i<sup>th</sup> demand_point to the j<sup>th</sup> supply point.
- **proximity.npy**: 100x4096 numpy array. A row *i* contains the sorted ids of the demand points based on the distance to the i<sup>th</sup> supply point.
- **reverse_proximity.npy**: 4096x100 numpy array. A row *i* contains the sorted ids of the supply points based on the distance to the i<sup>th</sup> demand_point.