# decaf_code

repo for decaf

Fair-GAD-Decaf/
│
├── env/
│   └── cfgad_env.yml                      # conda or pip environment
│
├── data/
│   ├── original/                          # untouched datasets (credit, bail, etc.)
│   └── injected/                          # fairness-injected versions
│       ├── credit/
│       ├── bail/
│       ├── german/
│       └── synthetic/
│
├── src/ 
│   ├── baseline.py                        # baseline eval code (with logging)
│   ├── generate.py                        # injection method for fairness (cf method)
│   ├── mod_gad/                           # your custom GAD models
│   ├── pygod_mod/                         # modified pygod version (if needed)
│   └── utils/                             # small helper functions
│
├── configs/                                   (记得洗一下)
│   ├── credit_params.yml
│   ├── bail_params.yml
│   ├── german_params.yml
│   ├── synthetic_params.yml
│  
│
├── results/
│   └── model_comparison_seaborn.png       # nice graphs
│
├── notebooks/
│   └── analysis_notebook.ipynb            # Optional: EDA or results summary
│
├── README.md
└── LICENSE

## env
The environment install:
```bash
conda env create -f environment/cfgad_env.yml
conda activate cfgad
```

## dataset

## outlier inject
outlier injection example:
```
python outlier_injection.py --dataset {dataset} --model {model} -outlier_type {outlier_type}
```
Argument | Options | Description
--dataset | credit, german, bail, synthetic | Dataset name
--model | anomalous, adone, cola, conad, dominant, dmgd, done, gaan, gadnr, gae, guide, ocgnn, one, radar, scan | Graph Anomaly Detection model
--outlier_type | structural, contextual, dice, path, cont_struc, path_dice | Type of outlier injection

## baseline eval


