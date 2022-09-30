The pipeline to produce a valid submission file is:

### 1. Generate chargers and distribution (Run genetic algorithm)
```
python generator.py 2019 -g {generations} -p {population}
python generator.py 2020 -g {generations} -p {population}
```
This will generate 4 files in the *outputs/* directory
- chargers_2019.csv
- chargers_2020.csv
- ds_2019.npy
- ds_2020.npy

### 2. Create the submission file
```
python submit.py
```
Creates a submission.csv file in the *outputs/* directory

### 3. Validate the submission file
```
python validate.py outputs/submission.csv
```
