import time
from functools import partial

import jax
import jax.numpy as jnp
import scipy.optimize
import jax.scipy.stats
import pandas as pd
from jax.scipy.stats import norm
from tqdm import tqdm

def setup_parameters():
    """Sets up the parameters for the exercise."""
    delta = jnp.array([2,2])
    Gamma = jnp.eye(2)
    sigma = jnp.array([[1, 0.5], [0.5, 1]])
    Ctilde = jnp.linalg.cholesky(sigma)
    eigvals, U = jnp.linalg.eigh(sigma)
    lambdas = eigvals
    C = Ctilde @ U
    b = C.T @ delta
    return delta, Gamma, sigma, Ctilde, C, b, lambdas
 
def crude_mc_2d(z, b, lambdas, x = 15):
    linear = jnp.dot(z, b)
    quadratic = jnp.sum(z * jnp.dot(z, jnp.diag(lambdas)), axis=1)
    X = (linear + quadratic) > x
    return jnp.mean(X), jnp.var(X, ddof=1)

def control_variate_2d(z, b, delta,lambdas, sigma, x=15):
    linear = jnp.dot(z, b)
    quadratic = jnp.sum(z * jnp.dot(z, jnp.diag(lambdas)), axis=1)
    X = ((linear + quadratic) > x).astype(jnp.float64)
    sigma_Q = jnp.sqrt(jnp.dot(delta, jnp.dot(sigma, delta)))
    
    Y = (linear > x).astype(jnp.float64)

    E_Y = 1 - jax.scipy.stats.norm.cdf(x / sigma_Q)
    Var_Y = E_Y * (1 - E_Y)
    cov_XY = jnp.cov(X, Y, ddof=1)[0, 1]

    c_star = -cov_XY / Var_Y
    _X = jnp.where(Var_Y > 0, X + c_star * (Y - E_Y), X)
    return jnp.mean(_X), jnp.var(_X, ddof=1)

def psi_L (theta, b, lambdas):
    term = (jnp.square(theta) * jnp.square(b)) / (1- 2 * theta * lambdas) - jnp.log(1 - 2 * theta * lambdas)
    return 0.5 * jnp.sum(term)

def psi_L_prime(theta, b, lambdas):
    term = jnp.sum((theta * jnp.square(b) * (1 - theta * lambdas)) / jnp.square(1 - 2 * theta * lambdas) + (lambdas) / (1 - 2 * theta * lambdas))
    return term

def importance_sampling_2d(z, theta_1, theta_2, b, lambdas, x=15):
    theta = jnp.array([theta_1, theta_2])
    z_tilt = z + theta

    likelihood_ratio = jnp.exp(-jnp.dot(z_tilt, theta) + 0.5 * jnp.sum(jnp.square(theta)))

    L = (
        b[0] * z_tilt[:, 0] + b[1] * z_tilt[:, 1] + (lambdas[0] * jnp.square(z_tilt[:, 0]) + lambdas[1] * jnp.square(z_tilt[:, 1]))
    )

    X = (L > x).astype(jnp.float64) * likelihood_ratio
    return jnp.mean(X), jnp.var(X, ddof=1)

def importance_sampling_2d_optimal(z, theta_star, b, lambdas, x=15):
    mu = (theta_star * b) / (1 - 2 * theta_star * lambdas)
    sigma = 1 / jnp.sqrt(1 - 2 * theta_star * lambdas)
    z_tilt = mu + z * sigma
    L = (
            b[0] * z_tilt[:, 0] + b[1] * z_tilt[:, 1] + (lambdas[0] * jnp.square(z_tilt[:, 0]) + lambdas[1] * jnp.square(z_tilt[:, 1]))
        )
    psi = psi_L(theta_star, b, lambdas)
    weight = jnp.exp(-theta_star * L + psi)

    X = (L > x).astype(jnp.float64) * weight
    return jnp.mean(X), jnp.var(X, ddof=1)

def main():
    x = 15
    delta, Gamma, sigma, Ctilde, C, b, lambdas = setup_parameters()

    max_lambda = jnp.max(lambdas)
    upper_bound = 1 / (2 * max_lambda) - 1e-5
    target = lambda theta: psi_L_prime(theta, b, lambdas) - x
    solution = scipy.optimize.root_scalar(target, bracket=[0, upper_bound], method='bisect')
    theta_star = solution.root

    variance_reduction_methods = {
        "Crude MC": partial(crude_mc_2d, b=b, lambdas=lambdas, x=x),
        "Control Variate": partial(control_variate_2d, b=b, delta=delta, lambdas=lambdas, sigma=sigma, x=x),
        "Importance Sampling (theta_1=0.5, theta_2=0.5)": partial(importance_sampling_2d, theta_1=0.5, theta_2=0.5, b=b, lambdas=lambdas, x=x),
        "Importance Sampling (theta=theta_star)": partial(importance_sampling_2d_optimal, theta_star=theta_star, b=b, lambdas=lambdas, x=x),
    }

    key = jax.random.PRNGKey(0)
    # JIT compilation and warming up
    for method in variance_reduction_methods:
        variance_reduction_methods[method] = jax.jit(variance_reduction_methods[method])
        # Running the funtions once to make sure that they are JIT compiled.
        z = jax.random.normal(key, (100, 2))
        _, _ = variance_reduction_methods[method](z)
    
    crude_estimator = jax.jit(partial(crude_mc_2d, b=b, lambdas=lambdas, x=x))
    crude_estimator(z)

    entries = []
    for n_sample in tqdm(
        jnp.linspace(200, 100_000, 250),
        desc="Running all methods for different sample sizes.",
    ):
        n_sample = int(n_sample)
        start_time = time.time()
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, (n_sample, 2))
        crude_mean, crude_var = jax.block_until_ready(crude_mc_2d(z, b=b, lambdas=lambdas, x=x))
        end_time = time.time()
        entries.append(
            dict(
                name="Crude MC",
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
            z = jax.random.normal(subkey, (n_sample, 2))
            mean_estimate, var_estimate = jax.block_until_ready(estimator(z))
            end_time = time.time()
            entries.append(
                dict(
                    name=method,
                    mean_estimate=mean_estimate,
                    var_estimate=var_estimate,
                    variance_reduction=crude_var / var_estimate,
                    n_sample=n_sample,
                    elapsed=end_time - start_time,
                )
            )
    df = pd.DataFrame.from_records(entries)
    df.to_csv("exercise_2_results.csv")
    print("DONE")

if __name__ == "__main__":
    main()