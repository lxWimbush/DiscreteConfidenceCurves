# Confidence Curves Library

This Python library provides a robust framework for generating and analyzing **confidence curves**. A confidence curve can be considered analogous to a stack of nested confidence intervals, and provides a representation of the **possibility** of different parameter values for a given statistical distribution, based on observed data. 

The **converted** confidence curves in this script correct the conservatism that is inherent in simply **direct** curves developed by simply stacking confidence intervals directly from the confidence box (c-box). They also allow for belief assignment that is protected from false confidence.

Note that calculation can become extremely expensive when the size of the observable sequence of outcomes gets very large. For example with the binomial case, it can take a few minutes at the moment to generate the relevant curve for `n=1000`. In this case, the difference between the direct and converted curve is likely to be so small that it may make more sense to simply use the direct curve for computational efficiency.

A test library of Singh plots is provided in `Testing.py`. Running this script should produce a series of valid Singh plots, and possibility values of endpoints that are less than `0.01` at each end. If this is not the case let me know!

---

## Features

* **Generalized `ConfidenceCurve` Base Class:** A flexible foundation for constructing confidence curves for various distributions.
* **Specialized Distribution Support:** Ready-to-use classes for:
    * **`Binomial`**: For estimating the probability of success ($p$) given observed successes ($k$) in trials ($n$).
    * **`Poisson`**: For estimating the rate parameter ($\lambda$) given an observed count ($k$).
    * **`NegativeBinomial`**: For estimating the probability of success ($p$) given observed trials ($n$) to reach a fixed number of successes ($k$).
    * **`Hypergeometric`**: For estimating the number of "successes" ($K$) in a finite population ($N$) given observed successes ($k$) in a sample ($n$).
    * **`ArrestedNegativeBinomial`**: For a more complex scenario where the process stops at either a maximum number of successes (`max_k`) or trials (`max_n`).
* **Possibility Calculation**: Determine the possibility of any specific parameter value or interval.
* **Alpha-Cut Generation**: Easily derive $\alpha$-cuts, which are equivalent to confidence intervals at a given possibility level $\alpha$.
* **Outward Rounding Precision**: Configurable precision for $\alpha$-cut calculations to ensure confidence intervals are robust.

---

## Installation

This library relies on `numpy` for numerical operations and `scipy` for statistical distributions and optimization.

1.  **Save the Code:** Copy the provided Python code into a file (e.g., `discocurves.py`).
2.  **Install Dependencies:** If you don't have them already, install `numpy` and `scipy` using pip:

    ```bash
    pip install numpy scipy
    ```

---

## Usage Examples

Each confidence curve is initialized with the **observed data** and any **fixed parameters** of the distribution. The `data` parameter for most classes is a single real number (e.g., an observed count or number of successes). For the `ArrestedNegativeBinomial` class, `data` is a tuple `(k, n)`.

Once a curve object is created, you can:
* Calculate the **possibility** of a specific parameter value or an interval of parameter values.
* Find **$\alpha$-cuts** (confidence intervals) at a desired possibility level.

Here are a few examples:

### 1. Binomial Confidence Curve

Estimate the probability of success ($p$) given observed successes ($k$) and total trials ($n$).

```python
from discocurves import Binomial

# Observed: 7 successes (k) in 10 trials (n)
# data = k (7), params = [n] ([10])
binomial_curve = Binomial(k=7, n=10)

# Calculate the possibility that p = 0.6
possibility_at_0_6 = binomial_curve.possibility(0.6)
print(f"Possibility of p=0.6: {possibility_at_0_6:.4f}")

# Get the 0.9-alpha cut (90% confidence interval)
lower_bound, upper_bound = binomial_curve.cut(0.9)
print(f"90% Confidence Interval for p: [{lower_bound:.4f}, {upper_bound:.4f}]")

# Possibility of an interval [0.5, 0.7]
possibility_interval = binomial_curve.possibility([0.5, 0.7])
print(f"Possibility of p in [0.5, 0.7]: {possibility_interval:.4f}")
```

### 2. Poisson Confidence Curve
Estimate the rate parameter ($\lambda$) given an observed count ($k$).

```python
from discocurves import Poisson

# Observed: 5 events (k)
# data = k (5), params = [] (empty list as no fixed parameters)
poisson_curve = Poisson(k=5)

# Calculate the possibility that lambda = 4.0
possibility_at_4_0 = poisson_curve.possibility(4.0)
print(f"Possibility of lambda=4.0: {possibility_at_4_0:.4f}")

# Get the 0.95-alpha cut (95% confidence interval)
lower_bound, upper_bound = poisson_curve.cut(0.95)
print(f"95% Confidence Interval for lambda: [{lower_bound:.4f}, {upper_bound:.4f}]")
```
### 3. Negative Binomial Confidence Curve
Estimate the probability of success ($p$) given a fixed number of successes ($k$) and the observed number of trials ($n$) required to achieve them.

```python
from discocurves import NegativeBinomial

# Observed: 12 trials (n) to get 5 successes (k)
# data = n (12), params = [k, max_n] ([5, calculated_max_n])
# Note: max_n is auto-calculated if not provided, but needs to be greater than n.
negative_binomial_curve = NegativeBinomial(k=5, n=12)

# Calculate the possibility that p = 0.4
possibility_at_0_4 = negative_binomial_curve.possibility(0.4)
print(f"Possibility of p=0.4 (Negative Binomial): {possibility_at_0_4:.4f}")

# Get the 0.9-alpha cut
lower_bound, upper_bound = negative_binomial_curve.cut(0.9)
print(f"90% Confidence Interval for p (Negative Binomial): [{lower_bound:.4f}, {upper_bound:.4f}]")
```

### 4. Arrested Negative Binomial Confidence Curve
This class is used when the process stops either after `max_k` successes or `max_n` trials. The data for this curve is a tuple `(k, n)`.

```python
from discocurves import ArrestedNegativeBinomial

# Observed: 3 successes (k) after 8 trials (n)
# Process stopped because max_k was 3 or max_n was 10. Here, max_k=3 was met.
# data = (k, n) = (3, 8), params = (max_k, max_n) = (3, 10)
arrested_nb_curve = ArrestedNegativeBinomial(k=3, n=8, max_k=3, max_n=10)

# Calculate the possibility that p = 0.35
possibility_at_0_35 = arrested_nb_curve.possibility(0.35)
print(f"Possibility of p=0.35 (Arrested NB): {possibility_at_0_35:.4f}")

# Get the 0.8-alpha cut
lower_bound, upper_bound = arrested_nb_curve.cut(0.8)
print(f"80% Confidence Interval for p (Arrested NB): [{lower_bound:.4f}, {upper_bound:.4f}]")
```

### 5. Hypergeometric Confidence Curve
Estimate the number of "successes" ($K$) in a finite population ($N$), given observed successes ($k$) in a sample of size ($n$).

```python
from discocurves import Hypergeometric

# Observed: 3 successes (k) in a sample of 10 (n) from a population of 50 (N)
# data = k (3), params = [N, n] ([50, 10])
hypergeometric_curve = Hypergeometric(k=3, N=50, n=10)

# Calculate the possibility that K = 15
possibility_at_15 = hypergeometric_curve.possibility(15)
print(f"Possibility of K=15 (Hypergeometric): {possibility_at_15:.4f}")

# Get the 0.85-alpha cut
lower_bound, upper_bound = hypergeometric_curve.cut(0.85)
print(f"85% Confidence Interval for K (Hypergeometric): [{lower_bound:.4f}, {upper_bound:.4f}]")
```

### 6. Other Distributions
If you can identify the points of intersection and the relevant c-boxes that you will need for a sequence of observable outcomes, you can pass these to the general `ConfidenceCurve` class and use it to make alpha cuts and assign possibility. 