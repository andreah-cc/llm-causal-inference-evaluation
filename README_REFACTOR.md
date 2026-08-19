# Refactored simulation package

This folder is structured for direct use in the GitHub repository.

- `notebooks/simulation_benchmark.ipynb`: readable benchmark/demo notebook
- `src/simulation/engine1.py` ... `engine8.py`: reusable DGP implementations
- `src/simulation/common.py`: shared simulation utilities
- `src/data_to_text.py`: data serialization and prompt utilities
- `src/evaluation.py`: bias/RMSE/coverage metrics
- `data/simulated/`: generated/example CSV output location

The refactor removes the Google Colab / Google Drive dependency and replaces global dependencies in Engine 5 with explicit parameters.
