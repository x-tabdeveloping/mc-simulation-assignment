import time
from functools import partial

import jax
import jax.numpy as jnp
import pandas as pd
from jax.scipy.stats import norm
from tqdm import tqdm

# Assume m=1, delta=2, gamma = 1, sigma = 1
# Estimating P(L>x)
# I denote ΔS as Z in this exercise, as it is a standard normal random variable and it's easier to type out in code :)


def closed_form(x=10):
    r1 = (-2 - jnp.sqrt(4 + 4 * x)) / 2
    r2 = (-2 + jnp.sqrt(4 + 4 * x)) / 2
    z_smaller_than_r1 = norm.cdf(r1)
    z_bigger_than_r2 = 1 - norm.cdf(r2)
    return jnp.where(x > -1, z_smaller_than_r1 + z_bigger_than_r2, 0)


def crude_mc(z, x=10):
    X = (2 * z + jnp.square(z)) > x
    return jnp.mean(X), jnp.var(X, ddof=1)


def control_variate_2z(z, x=10):
    """Estimates P(L>x) 2Z as a control variate."""
    X = (2 * z + jnp.square(z)) > x
    Y = (2 * z) > x
    E_Y = 1 - norm.cdf(x / 2)
    Var_Y = E_Y * (1 - E_Y)
    c_star = -jnp.cov(X, Y, ddof=1)[0, 1] / Var_Y
    # Falling back to Crude MC when the variance of the control variable is 0
    _X = jnp.where(Var_Y > 0, X + c_star * (Y - E_Y), X)
    return jnp.mean(_X), jnp.var(_X, ddof=1)


def control_variate_z_squared(z, x=10):
    """Estimates P(L>x) Z^2 as a control variate."""
    X = (2 * z + jnp.square(z)) > x
    Y = jnp.square(z) > x
    E_Y = jnp.where(x >= 0, 1 - norm.cdf(jnp.sqrt(x)), 1)
    Var_Y = E_Y * (1 - E_Y)
    c_star = -jnp.cov(X, Y, ddof=1)[0, 1] / Var_Y
    # Falling back to Crude MC when the variance of the control variable is 0
    _X = jnp.where(Var_Y > 0, X + c_star * (Y - E_Y), X)
    return jnp.mean(_X), jnp.var(_X, ddof=1)


def importance_sampling(z, x=10, exp_val=None):
    """Estimates P(L>x) using importance sampling.
    The change of measure (L*) is achieved by shifting Z's mean, such that E[L*] = x by default.
    You can change this value by specifying exp_val = <your value here>.
    """
    if exp_val is None:
        exp_val = x
    # Change of measure to where E[L*] = exp_val
    mu = -1 + jnp.sqrt(4 + 4 * exp_val) / 2
    z = z + mu
    likelihood_ratio = norm.pdf(z, loc=0, scale=1) / norm.pdf(z, loc=mu, scale=1)
    X = ((2 * z + jnp.square(z)) > x) * likelihood_ratio
    return jnp.mean(X), jnp.var(X, ddof=1)


def main():
    x = 10
    variance_reduction_methods = {
        "Control Variate (2Z)": partial(control_variate_2z, x=x),
        "Control Variate (Z^2)": partial(control_variate_z_squared, x=x),
        # I try this because there is more overlap with L in terms of the domain
        "Importance Sampling (E[L*] = 5)": partial(importance_sampling, exp_val=5, x=x),
        "Importance Sampling (E[L*] = x)": partial(importance_sampling, x=x),
    }
    key = jax.random.key(0)
    # JIT compilation and warming up
    for method in variance_reduction_methods:
        variance_reduction_methods[method] = jax.jit(variance_reduction_methods[method])
        # Running the funtions once to make sure that they are JIT compiled.
        z = jax.random.normal(key, 100)
        _, _ = variance_reduction_methods[method](z)
    # JIT compiling crude estimator
    crude_estimator = jax.jit(partial(crude_mc, x=x))
    crude_estimator(z)
    # Calculating actual result
    true_result = closed_form(x=x)
    entries = []
    for n_sample in tqdm(
        jnp.linspace(200, 100_000, 250),
        desc="Running all methods for different sample sizes.",
    ):
        n_sample = int(n_sample)
        start_time = time.time()
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (n_sample,))
        crude_mean, crude_var = jax.block_until_ready(crude_mc(z, x=x))
        end_time = time.time()
        entries.append(
            dict(
                name="Crude MC",
                true_result=true_result,
                mean_estimate=crude_mean,
                var_estimate=crude_var,
                variance_reduction=1,
                n_sample=n_sample,
                elapsed=end_time - start_time,
            )
        )
        for method in variance_reduction_methods:
            estimator = variance_reduction_methods[method]
            start_time = time.time()
            key, subkey = jax.random.split(key)
            z = jax.random.normal(subkey, (n_sample,))
            mean_estimate, var_estimate = jax.block_until_ready(estimator(z, x=x))
            end_time = time.time()
            entries.append(
                dict(
                    name=method,
                    true_result=true_result,
                    mean_estimate=mean_estimate,
                    var_estimate=var_estimate,
                    variance_reduction=crude_var / var_estimate,
                    n_sample=n_sample,
                    elapsed=end_time - start_time,
                )
            )
    df = pd.DataFrame.from_records(entries)
    df.to_csv("exercise_1_results.csv")
    print("DONE")


if __name__ == "__main__":
    main()
