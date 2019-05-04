# DateBase Host
In `database.py`
```python
    DATA_ROOT = '/home/fish/data/jdata'
```

# Adjustment parameters 
In `train.py`
```python
    param = {'learning_rate': 0.01, 'n_estimators': 1000, 'max_depth': 3,
             'min_child_weight': 5, 'gamma': 0, 'subsample': 1.0, 'colsample_bytree': 0.8,
             'scale_pos_weight': 1, 'eta': 0.3, 'silent': 1, 'objective': 'binary:logistic'}
    num_round = 283
```

# Run
Install requirement package
```bash
pip install requirsments.txt
```

Just
```bash
python train.py
```