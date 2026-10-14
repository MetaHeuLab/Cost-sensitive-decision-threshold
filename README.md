# Cost-sensitive decision-threshold and hyperparameter selection for customer churn prediction

Code for the manuscript *Joint Optimization of the Decision Threshold and Hyperparameters of
Gradient Boosting for Cost Reduction in Customer Churn Prediction*. It takes the KDD Cup 2009
Orange churn database and produces every table reported in the paper.


## Data

The database is not redistributed here. Download `KDDCup09_churn.arff` (50,000 customers, 230
anonymised variables, 7.34% churn) from the KDD Cup 2009 archive and pass its path with `--data`.
Nothing is written back to it.


## Run

```bash
python run_experiments.py --data KDDCup09_churn.arff     # fits everything, caches per fold
python make_tables.py     --data KDDCup09_churn.arff     # writes tables/table1 ... table10
python verify.py          --data KDDCup09_churn.arff     # compares against published values
```

`
```bash
python run_experiments.py --data ... --stage main         # Tables 3, 4A, 5, 7
python run_experiments.py --data ... --stage optimizers   # Table 6
python run_experiments.py --data ... --stage ablations    # Tables 4B and 10
python run_experiments.py --data ... --stage attribution  # Table 9
python run_experiments.py --data ... --learner CAT --folds 0 1 --time-limit 600
```


churn/
  config.py        seeds, costs and budgets; changing these changes the published numbers
  data.py          ARFF parsing, cleaning, 5x2 splits, fold-internal frequency encoding
  cost.py          cost function, exact cost-minimizing threshold, analytical threshold, metrics
  models.py        five learners, their search spaces and their default configurations
  optimizers.py    RIME, particle swarm, differential evolution, grey wolf
  experiment.py    one (learner, strategy) over the ten outer folds, cached per fold
  ablations.py     depth sweep, threshold transfer, cost-ratio sensitivity
  attribution.py   built-in, SHAP and permutation importance with rank stability
  statistics.py    Alpaydin's combined 5x2 cv F-test and the Holm correction
  tables.py        Tables 1 to 10 as CSV
run_experiments.py, make_tables.py, verify.py
```

## Method in brief

For hyperparameters `θ` and threshold `τ`, with `C_in` the cost on the inner selection set, the
framework minimises `F(θ) = min_τ C_in(θ, τ)`. The inner minimisation is solved exactly: sorting
the selection-set scores once and accumulating the confusion counts evaluates every candidate
threshold, so no part of the evaluation budget approximates a quantity available in closed form.
The outer minimisation over `θ` is carried out by the optimizer. The selected configuration is
refitted on the whole outer training half and applied once to the held-out half, with the
threshold chosen on the inner selection set.



## Licence

MIT, see `LICENSE`.
