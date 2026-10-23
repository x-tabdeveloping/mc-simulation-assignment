# mc-simulation-assignment
Assignment code for the Monte Carlo Simulation course at AU.

## Requirements
The following packages will have to be installed in your environment in order to be able to run our code

```bash
pip install plotly jax numpy pandas tqdm 
```

## First set of parameters (exercise a-e)

In order to produce results from the different Monte Carlo estimators, one has to run the `exercise_1.py` script, which then produces the `exercise_1_results.csv` file.

```bash
python3 exercise_1.py
```

In order to produce the plots used in the assignment text, the `produce_plots.py` script has to be used.
It should open the plots in new browser tabs.

```bash
python3 produce_plots.py
```

## Second set of parameters (exercise f-j)

Code implementations of the Monte Carlo estimators for the multivariate case can be found in the `exercise_2.ipynb` notebook.
