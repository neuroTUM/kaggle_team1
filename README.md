# Kaggle Team 1

## Initialization
Run
```sh
uv sync
uv run pre-commit install
```
to install all necessary packages.

## Slow potential features
```sh
uv run python main.py +experiment=slow_potential
```

## Time domain parameters
```sh
uv run python main.py feature_extraction.use=time_domain_parameters
```
