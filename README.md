# Misinformation detection

This repository contains code and research on automated misinformation detection.

## Setup

All code requires `python3.6+`

* (Optional) Create and activate a Python virtual environment
* In project root, run `make all`
* Update `config.json`:
    * `index`: `user`
    * `index`: `password`
    * These only need to be set if re-running the data preparation process is required

Wait for the makefile to complete (it downloads a 700MB archive with embeddings which, depending on your internet connection, may take a while).

Afterwards, you can start the app using `python run.py`. Jupyter notebooks can be browsed by executing `jupyter notebook`.

## Running the app

* Execute `jupyter notebook`, this will open a browser window
* Navigate to `notebooks/` directory in the browser window and open any notebook

In general, all python source codes are in the `src` directory -- this includes all machine learning models and data analysis code. Jupyter notebooks are used to execute the code in `src` and visualize the results in one place and are the best place to start when starting to use the code.

When starting a new notebook, copy the template in `notebooks/00_template.ipynb` -- the template loads config files and runs all necessary setup to be able to easily import sources from the `src` directory.

## Reproducing experiments

To re-run evaluation of all classification models tested, open and re-run `notebooks/02_misinfo_detection.ipynb`.

## Repository organization

* `data/`:
    * `data/interim/`: intermediate datasets (not final)
    * `data/processed/`: final processed data used in modelling, this directory mainly contain data splits
    * `data/raw/`: raw, unprocessed data files
* `models/`:
    * `models/config/`: model configuration files
    * `models/embeddings/`: embeddings
    * `models/trained/`: trained and serialized models and model logs and training logs
* `notebooks/`: jupyter notebooks
* `notes/`: various notes and documents related to the project
* `output/`:
    * `experiments/`: output from ML experiments
    * `output/figs/`: directory for plots
* `src/`: source code
* `config.example.json`: example project config
* `logging.example.json`: example logging config
* `Makefile`: project makefile
* `README.md`: this readme
* `requirements.txt`: Python dependencies
* `run.py`: main Python script for running the application
