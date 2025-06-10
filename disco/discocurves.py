import numpy as np
import scipy.stats as sps
from scipy.special import beta, digamma, betaln, factorial, binom
from scipy.optimize import root_scalar
from typing import Callable, Union, List, Tuple, Any, Iterable


class ConfidenceCurve():
    """Create a confidence curve from which alpha-cuts can be drawn, or
    possibility can be assigned to some hypothesis about the inferred parameter

    Parameters
    ----------
    data : (int | float)
        The single observed data point used to generate the confidence curve.
        For example, for a Binomial distribution, this would be the number of
        successes `k`; for Poisson, it would be the observed count.
    params : (list | tuple | np.ndarray)
        Known parameters of the distribution, which are held constant. For
        example in the binomial distribution, the sample size `n` would be a
        parameter. The exact contents depend on the specific ConfidenceCurve
        subclass.
    support : (list | tuple | np.ndarray)
        The support of the distribution for the inferred parameter. Either
        as a pair of interval limits (continuous support), or as a set of
        observable outcomes (discrete support) for the inferred parameter.
    alt_range : (list | tuple | np.ndarray)
        A sequence of alternative observable outcomes for the data that are
        considered when constructing the confidence curve. The observed data
        must be a member of this set. Its structure varies by subclass, for
        example the arrested negative binomial produces data that are pairs of
        values (k, n) rather than single discrete values.
    cdist : Callable[[Union[int, float], list | tuple | np.ndarray], tuple
    [scipy.stats.rv_continuous | scipy.stats.rv_discrete, scipy.stats.
    rv_continuous | scipy.stats.rv_discrete]]
        A function that takes the observed data value and known parameters
        and returns a pair of scipy.stats distributions. These distributions
        correspond to the lower and upper bounds of the confidence distribution
        for a given observation.
    intersections : np.ndarray
        An array of pre-calculated intersection points between the lower and
        upper bound distributions for various alternative outcomes in
        alt_range. These points form the "steps" of the confidence curve.
    max_poss_bounds : np.ndarray, optional
        An array of endpoints for the maximum possibility region (where
        possibility is 1) given the observed data. If `None`, it is calculated
        internally based on intersections and cutoffs.
    precision : float, optional
        The maximum allowable error for outward-directed rounding used when
        calculating alpha-cuts. This ensures that the calculated bounds
        are slightly wider than the true bounds, never narrower.
        Defaults to 1e-4.
    cutoffs : list, optional
        A list of two values, `[lower_cutoff, upper_cutoff]`, indicating
        where the confidence curve should be truncated within alt_range,
        respectively, to make computation feasible. Defaults to `[False, False]
        ` indicating that no truncation is needed.

    See Also
    --------
    Hypergeometric : A subclass for hypergeometric confidence curves.
    Poisson : A subclass for Poisson confidence curves.
    NegativeBinomial : A subclass for Negative Binomial confidence curves.
    Binomial : A subclass for Binomial confidence curves.
    ArrestedNegativeBinomial : A subclass for Arrested Negative Binomial
    confidence curves.
    """

    def __init__(
        self,
        data: Union[int, float],
        params: Union[List, Tuple, np.ndarray],
        support: Union[List, Tuple, np.ndarray],
        alt_range: Union[List, Tuple, np.ndarray],
        # Updated data type in Callable
        cdist: Callable[[Union[int, float], Union[List, Tuple, np.ndarray]], Tuple[sps.rv_continuous, sps.rv_discrete]],
        intersections: np.ndarray,
        max_poss_bounds: Union[np.ndarray, None] = None,
        precision: float = 1e-4,
        cutoffs: List[Any] = [False, False]
    ):
        self.data = data
        self.params = params
        self.support = support
        self.cdist = cdist
        self.precision = precision
        self.intersections = intersections
        self.alt_range = alt_range
        self.alt_count = len(self.alt_range)
        # Finds the position of the observed data within the alternative range.
        self.pos = np.where([A == self.data for A in self.alt_range])[0][0]
        self.cutoffs = cutoffs
        # Determine the bounds of the maximum possibility region (where
        # possibility is 1). This logic handles edge cases for 'pos' and
        # 'cutoffs' to set the correct indices.
        if max_poss_bounds is None:
            self.max_poss_bounds = self.intersections[[
                self.pos + (
                    (-1 if self.cutoffs[0] else 0) if self.pos == 0
                    else -2 if self.data == self.alt_range[-1]
                    else -1),
                self.pos + (
                    1 if self.pos == 0
                    else (0 if self.cutoffs[1] else -1)
                    if self.data == self.alt_range[-1]
                    else 0)]]
        else:
            self.max_poss_bounds = max_poss_bounds
        # Caches for previously calculated possibility values and alpha cuts.
        self.results = {}
        self.cuts = {}
        # Pre-calculates possibility values for the intersection points.
        self.steps = self.possibility(self.intersections)
        # check if continuous. If support is two end points, only possible
        # discrete support is similar binary
        self.continuous = len(self.support) == 2

    def possibility(self, theta):
        """Calculate the possibility of a given theta, either on a pointwise basis if theta is a scalar, over each value of theta if it is an iterable of length >2, or as an interval if it is an iterable of length 2.

        Parameters
        ----------
        theta: (Iterable, float)
            Either a single value for which a possibility is to be calculated, or an iterable. If an iterable of length 2, this is interpreted as
            an interval, and the possibiltiy is calculated as such. Otherwise a
            possibility is calculated for each element in the iterable on a
            pointwise basis.

        Returns
        ----------
        poss: (float, np.ndarray)
            The calculated possibility for the provided theta. Either as a single value, if theta was itself a scalar or a pair of scalars, or as an array of values otherwise.
        """
        if isinstance(theta, (list, tuple, np.ndarray)):
            # If theta is a pair of endpoints, evaluate as an interval,
            # otherwise evaluate as a set of individual points
            if len(theta) == 2:
                # Determine the position of theta's endpoints relative to the
                # intersections.
                theta_pos = [np.sum(self.intersections < t) for t in theta]
                # If the observed data's position falls within the maximum
                # possibility interval, possibility is 1.
                if theta_pos[0] <= self.pos <= theta_pos[1]:
                    poss = np.array([1])
                # Otherwise, the possibility of the interval is the maximum
                # possibility of its endpoints.
                else:
                    poss = max(
                        self.possibility(theta[0]),
                        self.possibility(theta[1])
                    )
            else:
                # Recursively calculate possibility for each point in the
                # iterable.
                poss = np.hstack([
                    self.possibility(t) for t in theta
                ])
        else:  # theta is a scalar
            if theta in self.results:
                # Return cached result if available.
                poss = self.results[theta]
            else:
                # Determine theta's position relative to the intersection
                # points.
                theta_pos = np.sum(self.intersections <= theta)
                # If theta is to the left of the max possibility region.
                if theta < self.max_poss_bounds[0]:
                    if self.cutoffs[0] and theta < self.intersections[0]:
                        # Handle lower cutoff (truncation).
                        poss = self.possibility(self.intersections[0])
                    else:
                        # Calculate possibility using the lower bound
                        # distribution (CDF) and potentially the upper bound
                        # distribution (survival function) of the adjacent
                        # alternative outcome.
                        poss = self.cdist(
                            self.data,
                            self.params
                        )[0].cdf(theta)
                        if theta >= self.intersections[0]:
                            poss += self.cdist(
                                self.alt_range[theta_pos - 1],
                                self.params
                            )[1].sf(theta)
                elif (
                    self.max_poss_bounds[0]
                    <= theta
                        <= self.max_poss_bounds[1]):
                    # If theta is within the maximum possibility region,
                    # possibility is 1.
                    poss = 1
                else:
                    # If theta is to the right of the max possibility region.
                    if self.cutoffs[1] and theta > self.intersections[-1]:
                        # Handle upper cutoff (truncation).
                        poss = self.possibility(self.intersections[0])
                    else:
                        # Calculate possibility using the upper bound
                        # distribution (survival function) and potentially the
                        # lower bound distribution (CDF) of the adjacent
                        # alternative outcome.
                        poss = self.cdist(
                            self.data,
                            self.params
                        )[1].sf(theta)
                        if theta < self.intersections[-1]:
                            poss += self.cdist(
                                self.alt_range[
                                    theta_pos
                                    + (0 if self.data == self.alt_range[0]
                                       else 1)],
                                self.params
                            )[0].cdf(theta)
                # Cache the calculated possibility for future use.
                self.results[theta] = poss
        return poss

    def cut(self, alpha: float) -> Tuple[float, float]:
        """Calculate an alpha-cut from the confidence curve.

        An alpha-cut defines an interval where every value by definition has a
        possibility value greater than or equal to `alpha`. The bounds are
        calculated with outward rounding to ensure the true interval is
        contained within the returned bounds, considering the `precision`.

        Parameters
        ----------
        alpha : float
            A scalar in the range `[0, 1]` representing the alpha level of the cut.

        Returns
        -------
        Lbound, Rbound : (float, float)
            A pair of floats representing the lower and upper bounds of the
            alpha-cut interval. These bounds are calculated with outward rounding
            according to `self.precision`. For example, if `precision` is `1e-4`,
            the upper bound could be up to `1e-4` greater than the true upper bound,
            but never lower. Similarly, the lower bound could be up to `1e-4` lower
            than the true lower bound, but never greater.
        """
        if alpha in self.cuts:
            # Return cached cut if available.
            Lbound, Rbound = self.cuts[alpha]
        else:
            # Calculate Left Bound
            if (self.cutoffs[0] and self.possibility(
                    self.intersections[0]) >= alpha) or self.pos == 0:
                # Check if the lower bound of the support can be used as Lbound
                Lbound = self.support[0]
            else:
                # Find the segment where the alpha-cut boundary lies.
                pos = sum(self.steps[:self.pos] <= alpha)
                # Define function for root finding: possibility(x) - alpha = 0.
                def fun(x): return self.possibility(x) - alpha
                # Set up the bracket for root_scalar based on intersection
                # points.
                bracket = [
                    0 if pos == 0 else self.intersections[pos - 1],
                    self.intersections[pos]
                ]
                # Use root_scalar to find the left boundary of the alpha-cut.
                Lbound = root_scalar(
                    fun,
                    bracket=bracket,
                    x0=np.mean(bracket),
                    xtol=self.precision
                ).root

            # Calculate Right Bound
            if ((self.cutoffs[1] and self.possibility(
                    self.intersections[-1]) >= alpha)
                    or self.pos == len(self.intersections) - 1):
                # Check if the upper bound of the support can be used as Rbound
                Rbound = self.support[1]
            else:
                # Find the segment where the alpha-cut boundary lies.
                pos = sum(self.steps[self.pos:] <= alpha)
                # Define function for root finding: possibility(x) - alpha = 0.
                def fun(x): return self.possibility(x) - alpha
                # Set up the bracket for root_scalar based on intersection
                # points.
                bracket = [
                    self.intersections[-pos - 1],
                    # Adjusted bracket for clarity
                    self.intersections[-pos] if pos > 0 else self.support[-1]
                ]
                # Use root_scalar to find the right boundary of the alpha-cut.
                Rbound = root_scalar(
                    fun,
                    bracket=bracket,
                    x0=np.mean(bracket),
                    xtol=self.precision
                ).root

            # Adjust bounds based on continuous/discrete support and precision
            # for outward rounding.
            if not self.continuous:
                Lbound = np.floor(Lbound)
                Rbound = np.ceil(Rbound)
            else:
                Lbound -= self.precision / 2
                Rbound += self.precision / 2
        # Cache the calculated bounds.
        self.cuts[alpha] = (Lbound, Rbound)
        return Lbound, Rbound


def _hg_pderiv(N: int, n: int, K: np.ndarray, k: int, e: int) -> np.ndarray:
    """Calculates the derivative of the log-likelihood for the Hypergeometric distribution with respect to the parameter K (population size).

    Parameters
    ----------
    N : int
        Total population size (fixed parameter).
    n : int
        Sample size (fixed parameter).
    K : np.ndarray
        Array of possible values for the population size (inferred parameter).
    k : int
        Number of successes in the sample (observed data).
    e : int
        An offset (0 or 1) used to evaluate the derivative at `K-e` or `K+e`.
        Typically 0 for one bound and 1 for the other.

    Returns
    -------
    np.ndarray
        Array of derivative values for each K in the input array.
    """
    return (
        digamma(K - k - e + 1) -
        digamma(N - n - K + k + e + 1) -
        digamma(k + e) +
        digamma(n - k + 1 - e)
    )


def _hg_max_poss(N: int,
                 k: int,
                 n: int,
                 K: Union[np.ndarray,
                          None] = None) -> Tuple[int,
                                                 int]:
    """Calculates the boundaries of the maximum possibility region (possibility = 1) for the Hypergeometric distribution's confidence curve.

    Parameters
    ----------
    N : int
        Total population size (fixed parameter).
    k : int
        Number of successes in the sample (observed data).
    n : int
        Sample size (fixed parameter).
    K : np.ndarray, optional
        An array of possible values for the population size `K`. If `None`,
        it defaults to `np.arange(k, N - (n - k) + 1)`.

    Returns
    -------
    (int, int)
        A tuple containing the lower and upper bounds of the maximum possibility region.
    """
    if K is None:
        K = np.arange(k, N - (n - k) + 1)
    # The bounds are found where the derivative changes sign.
    # The conditions (k > 0 and k < n) handle edge cases where min/max k are
    # observed.
    return (
        np.where(_hg_pderiv(N, n, K, k, 0) >= 0)[0][0] + k if k > 0
        # If k=0, lower bound is 0 (no successes implies no successes in
        # population K)
        else 0,
        np.where(_hg_pderiv(N, n, K, k, 1) < 0)[0][-1] + k if k < n
        # If k=n, upper bound is N (all successes implies all successes in
        # population K)
        else N
    )


def _hg_intersection(
        N: int,
        n: int,
        K_support: np.ndarray,
        k0: int,
        k1: int,
        e: int) -> np.ndarray:
    """Calculates the ratio of binomial coefficients and beta functions for
    finding intersection points in the Hypergeometric confidence curve.

    Parameters
    ----------
    N : int
        Total population size.
    n : int
        Sample size.
    K_support : np.ndarray
        Array of possible values for the population size (inferred parameter).
    k0 : int
        Observed number of successes.
    k1 : int
        Alternative number of successes (alternative outcome).
    e : int
        An offset (0 or 1) used in the formula to differentiate between
        lower and upper intersection calculations.

    Returns
    -------
    np.ndarray
        An array of values representing the ratio, used to find intersection points.
    """
    return binom(N - n,
                 K_support - k0 - e) / binom(N - n,
                                             K_support - k1 - (1 - e)) * beta(k1 + (1 - e),
                                                                              n - k1 + e) / beta(k0 + e,
                                                                                                 n - k0 + (1 - e))


class Hypergeometric(ConfidenceCurve):
    """Represents a Confidence Curve for the Hypergeometric distribution.

    This class extends `ConfidenceCurve` to specifically handle the
    confidence curve for the population size (K) in a hypergeometric setting,
    given observed `k` successes in a sample of size `n` from a total
    population `N`. The inferred parameter is typically the total number of
    items of a certain type in the population (K).

    Parameters
    ----------
    k : int
        The observed number of successes in the sample (data).
    N : int
        The total population size (fixed parameter).
    n : int
        The sample size (fixed parameter).
    support : np.ndarray, optional
        The support for the inferred parameter (K), i.e., possible values for K.
        Defaults to `np.arange(k, N - (n - k) + 1)`.
    alt_range : np.ndarray, optional
        Alternative observable `k` values used to construct the curve.
        Defaults to `np.arange(max(0, k - N + n), min(n, N - n + k) + 1)`.
    cdist : Callable, optional
        A function returning the lower and upper bounds of the c-box.
    precision : float, optional
        The maximum allowable error for outward-directed rounding. Defaults to
        `1e-4`.
    intersections : np.ndarray, optional
        Pre-calculated intersection points for the imprecise likelihoods. If
        `None`, they are calculated internally using `_hg_intersection` and
        `_hg_max_poss`.
    """

    def __init__(self, k: int, N: int, n: int, support: Union[np.ndarray, None] = None,
                 alt_range: Union[np.ndarray, None] = None,
                 cdist: Union[Callable[[int,
                                        List[int]],
                                       Tuple[sps.rv_continuous,
                                             sps.rv_discrete]],
                              None] = None,
                 # `data` is int, `params` is list of ints
                 precision: float = 0.0001, intersections: Union[np.ndarray, None] = None):

        # Set default support if not provided. This defines the range of the
        # inferred parameter K.
        if support is None:
            support = np.arange(k, N - (n - k) + 1)
        # Set default alternative range if not provided. These are alternative
        # `k` values used in the curve construction.
        if alt_range is None:
            alt_range = np.arange(max(0, k - N + n), min(n, N - n + k) + 1)
        # Set default cdist function if not provided. This defines the
        # distributions used for the c-box bounds.
        if cdist is None:
            def cdist(data: int, params: List[int]): return [
                sps.betabinom(params[0] - params[1], data,
                              params[1] - data + 1, loc=data),
                sps.betabinom(params[0] - params[1], data + 1,
                              params[1] - data, loc=data + 1),
            ]
        # Calculate intersections and maximum possibility bounds if not
        # provided.
        if intersections is None:
            intersections = []
            # Intersections for k0 < k (left side of the max possibility
            # region)
            for k0 in range(max(0, k - (N - n)) + (0 if k == n else 0), k - 1):
                intersections += [
                    np.where(
                        _hg_intersection(N, n, support, k, k0, 0) < 1)[0][-1]
                    + k
                ]
            # Add the maximum possibility region boundaries.
            intersections += [*_hg_max_poss(N, k, n, np.arange(k, N + 1))]
            max_poss_bounds = [intersections[-2], intersections[-1]]
            # Intersections for k0 > k (right side of the max possibility
            # region)
            for k0 in range(k + 2, min(n, N - (n - k)) + (1 if k > 0 else 0)):
                intersections += [
                    np.where(
                        _hg_intersection(N, n, support, k, k0, 1) >= 1)[0][-1]
                    + k
                ]
            intersections = np.array(intersections)
        # Call the parent class constructor
        super().__init__(k, [N, n],
                         support=support,
                         alt_range=alt_range,
                         cdist=cdist,
                         intersections=intersections,
                         max_poss_bounds=max_poss_bounds,
                         precision=precision)


def _poi_intersection(k0: int, k1: int) -> float:
    """Calculates the intersection point of the imprecise likelihood functions
    for two Poisson distributions with observed counts `k0` and `k1`.

    Parameters
    ----------
    k0 : int
        Observed count.
    k1 : int
        Alternative count.

    Returns
    -------
    float
        The intersection point (rate parameter).
    """
    return (factorial(k1) / factorial(k0 - 1))**(1 / (k1 - k0 + 1))


class Poisson(ConfidenceCurve):
    """Represents a Confidence Curve for the Poisson distribution.

    This class extends `ConfidenceCurve` to model the confidence curve for the
    rate parameter (lambda) of a Poisson distribution, given an observed
    count `k`.

    Parameters
    ----------
    k : int
        The observed count (data).
    max_k : int, optional
        The maximum `k` value to consider in the `alt_range`. If `None`,
        defaults to `10 * max(1, k)`. This is necessary to make calculation
        feasible.
    support : np.ndarray, optional
        The support for the inferred rate parameter (lambda).
        Defaults to `np.array([0, np.inf])`.
    alt_range : np.ndarray, optional
        Alternative observable `k` values used to construct the curve.
        Defaults to a range up to `max_k`.
    cdist : Callable, optional
        A function returning the lower and upper bound c-box distributions.
        Defaults to a specific implementation for the Poisson case.
    precision : float, optional
        The maximum allowable error for outward-directed rounding. Defaults to
        `1e-4`.
    intersections : np.ndarray, optional
        Pre-calculated intersection points for the imprecise likelihoods. If `None`,
        they are calculated internally using `_poi_intersection`.
    """

    def __init__(self, k: int, max_k: Union[int, None] = None, support: Union[np.ndarray, None] = None,
                 alt_range: Union[np.ndarray, None] = None,
                 cdist: Union[Callable[[int,
                                        List[Any]],
                                       Tuple[sps.rv_continuous,
                                             sps.rv_discrete]],
                              None] = None,
                 # `data` is int, `params` is list of Any (empty)
                 precision: float = 0.0001, intersections: Union[np.ndarray, None] = None):

        # Set default max_k if not provided.
        if max_k is None:
            max_k = 10 * (max(1, k))
        # Set default support for the Poisson rate (lambda), which is
        # continuous from 0 to infinity.
        if support is None:
            support = np.array([0, np.inf])
        # Set default alternative range if not provided. Includes counts up to
        # max_k, with infinity only added as required by the algorithm.
        if alt_range is None:
            alt_range = np.concatenate((np.arange(max_k + 1), [np.inf]))
        # Set default cdist function if not provided. Uses Chi-squared
        # distributions.
        if cdist is None:
            def cdist(data: int, params: List[Any]): return [
                sps.chi2(df=2 * data + 1e-21, scale=0.5),
                sps.chi2(df=2 * (data + 1), scale=0.5)
            ]
        # Calculate intersections if not provided.
        if intersections is None:
            intersections = np.array(
                # Intersections for k0 < k
                [_poi_intersection(k, i) for i in range(k - 1)] +
                # Maximum possibility region bounds
                [np.exp(digamma(k)), np.exp(digamma(k + 1))] +
                # Intersections for k0 > k
                [_poi_intersection(k + 1, i) for i in range(k + 1, max_k + 1)]
            )
        # Call the parent class constructor.
        # Note: params is an empty list as Poisson has no additional fixed
        # parameters in this setup.
        super().__init__(k, [],
                         support=support,
                         alt_range=alt_range,
                         cdist=cdist,
                         intersections=intersections,
                         precision=precision,
                         # Lower cutoff False, upper cutoff related to max_k
                         cutoffs=[False, max_k])


def _nb_intersection(k: int, ni: int, nj: int) -> float:
    """Calculates the intersection point for two Negative Binomial imprecise
    likelihoods with observed successes `k` and different numbers of trials
    (`ni`, `nj`).

    This helper function is used in constructing the Negative Binomial
    confidence curve.

    Parameters
    ----------
    k : int
        The number of successes (fixed parameter for the distribution).
    ni : int
        Observed number of trials.
    nj : int
        Alternative number of trials

    Returns
    -------
    float
        The intersection point (probability `p`).
    """
    return 1 - (
        beta(k, ni - k + (1 if nj > ni else 0)) /
        beta(k, nj - k + (1 if nj < ni else 0))
    )**(1 / (ni - nj + ((1 if nj > ni else 0) - (1 if nj < ni else 0))))


def _nb_max_poss(k: int, n: int) -> float:
    """Calculates a boundary of the maximum possibility region for the Negative Binomial confidence curve for the probability of success `p`.

    Note that this would be implemented with two calls to this function, one for each bound.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.

    Returns
    -------
    float
        A boundary value for the probability `p` where possibility is maximized.
    """
    # Related to the mode of the posterior Beta distribution for the Negative
    # Binomial.
    return 1 - np.exp(digamma(n - k)) / np.exp(digamma(n))


class NegativeBinomial(ConfidenceCurve):
    """Represents a Confidence Curve for the Negative Binomial distribution.

    This class extends `ConfidenceCurve` to model the confidence curve for the
    probability of success (`p`) in a Negative Binomial distribution, given
    `k` successes and `n` total trials. The inferred parameter is typically `p`.

    Parameters
    ----------
    k : int
        The number of successes (fixed parameter of the NB distribution).
    n : int
        The observed number of trials (data).
    max_n : int, optional
        The maximum `n` value to consider in the `alt_range`. If `None`,
        defaults to `10 * max(1, n)`. This is necessary to make calculation
        feasible.
    support : np.ndarray, optional
        The support for the inferred probability `p`. Defaults to `np.array([0,
        1])`.
    alt_range : np.ndarray, optional
        Alternative observable `n` values used to construct the curve.
        Defaults to a reversed range from `k` to `max_n`.
    cdist : Callable, optional
        A function returning the lower and upper bound Beta distributions of
        the c-box. Defaults to a specific implementation for the Negative
        Binomial case.
    precision : float, optional
        The maximum allowable error for outward-directed rounding. Defaults to
        `1e-4`.
    intersections : np.ndarray, optional
        Pre-calculated intersection points for the confidence curve. If `None`,
        they are calculated internally using `_nb_intersection` and
        `_nb_max_poss`.
    """

    def __init__(self, k: int, n: int, max_n: Union[int, None] = None, support: Union[np.ndarray, None] = None,
                 alt_range: Union[np.ndarray, None] = None,
                 cdist: Union[Callable[[int,
                                        List[int]],
                                       Tuple[sps.rv_continuous,
                                             sps.rv_discrete]],
                              None] = None,
                 # `data` is int, `params` is list of int
                 precision: float = 0.0001, intersections: Union[np.ndarray, None] = None):

        # Set default max_n and assert that it must be greater than n if
        # provided.
        if max_n is None:
            max_n = 10 * (max(1, n))
        else:
            assert max_n != n, "Max_n must be larger than n"

        # Set default support for the probability p.
        if support is None:
            support = np.array([0, 1])
        # Set default alternative range for n (trials). Reversed to impose
        # ordering with initial values representing those that correspond to
        # lower inferred probabilities.
        if alt_range is None:
            alt_range = np.arange(k, max_n + 1)[::-1]
        # Set default cdist function if not provided. Uses Beta distributions
        # for bounds.
        if cdist is None:
            def cdist(data: int, params: List[int]): return [
                sps.beta(params[0], data - params[0] + 1),  # Upper/Left bound
                sps.beta(params[0], data - params[0])  # Lower/Right bound
            ]
        # Calculate intersections and maximum possibility bounds if not
        # provided.
        if intersections is None:
            intersections = [
                _nb_intersection(
                    k, n, i) for i in range(
                    k, n - 1)]  # Intersections for ni < n
            # Maximum possibility bounds
            intersections += [_nb_max_poss(k, n), _nb_max_poss(k, n + 1)]
            max_poss_bounds = [intersections[-1], intersections[-2]]
            # Intersections for ni > n
            intersections += [
                _nb_intersection(k, n, i)
                for i in range(n + 2, max_n + (1 if k < n else 1))
            ]
            # Ensure intersections are sorted for correct indexing
            intersections = np.sort(intersections)
        # Call the parent class constructor.
        super().__init__(n,
                         [k, max_n],
                         support=support,
                         alt_range=alt_range,
                         cdist=cdist,
                         intersections=intersections,
                         precision=precision,
                         max_poss_bounds=max_poss_bounds,
                         cutoffs=[max_n, False])


def _bin_intersection(k: int, n: int, i: int) -> float:
    """Calculates the intersection point for two Binomial imprecise likelihoods
    with different observed successes `k` and `i`, given `n` trials.

    Parameters
    ----------
    k : int
        Observed number of successes.
    n : int
        Total number of trials (fixed parameter).
    i : int
        Alternative number of successes.

    Returns
    -------
    float
        The intersection point (probability `p`).
    """
    return 1 / (1 + np.exp(-(1 / (k - i - 1)) *
                            (betaln(k, n - k + 1) - betaln(i + 1, n - i))))


def _bin_max_poss(k: int, n: int) -> float:
    """Calculates a boundary of the maximum possibility region for the Binomial
    confidence curve given observed data.

    Again, this would usually be called twice, once for each bound.

    Parameters
    ----------
    k : int
        Number of successes.
    n : int
        Number of trials.

    Returns
    -------
    float
        A boundary value for the probability `p` where possibility is maximized.
    """
    return 1 / (1 + np.exp(digamma(n - k + 1) - digamma(k)))


class Binomial(ConfidenceCurve):
    """Represents a Confidence Curve for the Binomial distribution.

    This class extends `ConfidenceCurve` to model the confidence curve for the
    probability of success (`p`) in a Binomial distribution, given `k` successes
    in `n` trials.

    Parameters
    ----------
    k : int
        The observed number of successes (data).
    n : int
        The total number of trials (fixed parameter).
    support : np.ndarray, optional
        The support for the inferred probability `p`. Defaults to `np.array([0, 1])`.
    alt_range : np.ndarray, optional
        Alternative observable `k` values used to construct the curve.
        Defaults to `np.arange(n + 1)`.
    cdist : Callable, optional
        A function returning the lower and upper bound Beta distributions of the c-box. Defaults to a specific implementation for the Binomial case.
    intersections : np.ndarray, optional
        Pre-calculated intersection points for the confidence curve. If `None`,
        they are calculated internally using `_bin_intersection` and
        `_bin_max_poss`.
    precision : float, optional
        The maximum allowable error for outward-directed rounding. Defaults to
        `1e-4`.
    """

    def __init__(self, k: int, n: int, support: Union[np.ndarray, None] = None, alt_range: Union[np.ndarray, None] = None,
                 cdist: Union[Callable[[int,
                                        List[int]],
                                       Tuple[sps.rv_continuous,
                                             sps.rv_discrete]],
                              None] = None,
                 # `data` is int, `params` is list of int
                 intersections: Union[np.ndarray, None] = None, precision: float = 0.0001):

        # Set default support for the probability p.
        if support is None:
            support = np.array([0, 1])
        # Set default alternative range for k (successes).
        if alt_range is None:
            alt_range = np.arange(n + 1)
        # Set default cdist function if not provided. Uses Beta distributions
        # for bounds of c-box.
        if cdist is None:
            def cdist(data: int, params: List[int]): return [
                sps.beta(
                    data + (1 if data == 0 else 0),
                    params[0] - data + 1, loc=1 if data == 0 else 0
                ),  # Upper/Left bound
                sps.beta(
                    data + 1,
                    params[0] - data + (1 if data == params[0]else 0),
                    loc=-1 if data == params[0] else 0
                )  # Lower, Right bound
            ]
        # Calculate intersections and maximum possibility bounds if not
        # provided.
        if intersections is None:
            intersections = [
                _bin_intersection(
                    k, n, i) for i in range(
                    k - 1)]  # Intersections for i < k
            # Maximum possibility bounds
            intersections += [_bin_max_poss(k, n), _bin_max_poss(k + 1, n)]
            # Intersections for i > k
            intersections += [_bin_intersection(i, n, k)
                              for i in range(k + 2, n + 1)]
            intersections = np.array(intersections)
        # Determine max_poss_bounds from the calculated intersections.
        max_poss_bounds = [intersections[k - 1 if k > 0 else k],
                           intersections[k if k > 0 else k + 1]]
        # Call the parent class constructor.
        super().__init__(k,
                         [n],
                         support=support,
                         alt_range=alt_range,
                         cdist=cdist,
                         intersections=intersections,
                         max_poss_bounds=max_poss_bounds,
                         precision=precision)


class ArrestedNegativeBinomial(ConfidenceCurve):
    """Represents a Confidence Curve for the Arrested Negative Binomial distribution.

    This class extends `ConfidenceCurve` to handle the confidence curve for a
    more complex scenario involving an "arrested" negative binomial process,
    where observations stop either after `max_k` successes or `max_n` trials,
    whichever comes first.

    Parameters
    ----------
    k : int
        The observed number of successes.
    n : int
        The observed number of trials.
    max_k : int
        The maximum number of successes allowed before the process stops.
    max_n : int
        The maximum number of trials allowed before the process stops.
    support : np.ndarray, optional
        The support for the inferred probability `p`. Defaults to `np.array([0,
        1])`.
    alt_range : list, optional
        Alternative observable `(k, n)` pairs that define the curve.
        Defaults to a specific structure combining ranges for `k` and `n`.
    cdist : Callable, optional
        A function returning the lower and upper bound Beta distributions for
        the c-box. Defaults to a specific implementation for the Arrested
        Negative Binomial case,
        handling conditions based on observed `k` relative to `max_k`.
    precision : float, optional
        The maximum allowable error for outward-directed rounding. Defaults to
        `1e-4`.
    intersections : np.ndarray, optional
        Pre-calculated intersection points for the confidence curve. If `None`,
        they are calculated internally based on `_nb_intersection`,
        `_bin_intersection`, `_nb_max_poss`, `_bin_max_poss`, and `root_scalar`
        for more complex intersections.
    """

    def __init__(self, k: int, n: int, max_k: int, max_n: int, support: Union[np.ndarray, None] = None, alt_range: Union[List[List[int]], None] = None,
                 cdist: Union[Callable[[Tuple[int,
                                              int],
                                        Tuple[int,
                                              int]],
                                       Tuple[sps.rv_continuous,
                                             sps.rv_discrete]],
                              None] = None,
                 # `data` is tuple, `params` is tuple
                 precision: float = 0.0001, intersections: Union[np.ndarray, None] = None):

        # Assert that either k or n equals their respective max values, as per
        # "arrested" definition.
        assert max_k == k or max_n == n, "Either max_k=k or max_n=n based on the arrested process."

        # Set default support for the probability p.
        if support is None:
            support = np.array([0, 1])
        # Set default alternative range if not provided. This involves iterating through
        # possible (k, n) pairs that define the boundary conditions of the
        # arrested process.
        if alt_range is None:
            # Cases where n hits max_n first
            alt_range = [[i, max_n] for i in range(max_k)]
            # Cases where k hits max_k first (reversed)
            alt_range += [[max_k, i] for i in range(max_k, max_n + 1)[::-1]]
        # Set default cdist function. It returns Beta distributions based on
        # which condition (k=max_k or n=max_n) was met first for the observed
        # data.
        if cdist is None:
            def cdist(data: Tuple[int, int], params: Tuple[int, int]): return [
                # If maxk==k, use the negative binomial c-box as the base
                sps.beta(data[0], data[1] - data[0] + 1),
                sps.beta(data[0], data[1] - data[0])
            ] if data[0] == params[0] else [
                # Otherwise, use the binomial c-box as the base
                sps.beta(data[0], data[1] - data[0] + 1),
                sps.beta(data[0] + 1, data[1] - data[0])
            ]
        # Calculate intersections and maximum possibility bounds if not
        # provided.
        if intersections is None:
            # If the observation hit the max_k boundary, NB is the base
            if k == max_k:
                # Use NB intersection for relevant alt_range items
                intersections = [
                    _nb_intersection(k, n, d[1])
                    for d in alt_range[k + (max_n - n) + 2:][::-1]
                ]
                # NB max possibility bounds
                intersections += [
                    _nb_max_poss(k, n), _nb_max_poss(k, n + 1)
                    if n < max_n
                    # If n is at max_n, use Binomial max_poss for that bound
                    else _bin_max_poss(k, n)
                ]
                max_poss_bounds = [intersections[-1], intersections[-2]]
                intersections += [
                    _nb_intersection(k, n, d[1])
                    for d in alt_range[max_k:k + (max_n - n) - 1][::-1]
                ]

                def _intfun(x: float, *args: Tuple[int, int]) -> float:
                    """Helper function for root finding specific to k == max_k case."""
                    return (
                        cdist(args,
                              # Upper bound PDF for alternative
                              (max_k, max_n))[1].pdf(x)
                        - cdist((k, n),
                                # Lower bound PDF for observed
                                (max_k, max_n))[0].pdf(x)
                    )
                # Find intersections through root finding
                intersections += [
                    root_scalar(
                        _intfun,
                        args=tuple(d),
                        bracket=[
                            1e-12,  # Small positive lower bound for bracket
                            intersections[-1]]
                    ).root
                    for d in alt_range[:k - (1 if n == max_n else 0)][::-1]
                ]
                intersections = np.array(intersections[::-1])
            else:
                # If the observation hit the max_n boundary, binomial is the
                # base
                # Use Binomial intersection
                intersections = [
                    _bin_intersection(
                        k, n, d[0]) for d in alt_range[:k - 1]
                ] if k > 0 else []
                intersections += [
                    _bin_max_poss(k, n),
                    _bin_max_poss(k + 1, n)
                ]
                # Binomial max possibility bounds
                max_poss_bounds = [intersections[-2], intersections[-1]]
                intersections += [
                    _bin_intersection(d[0], n, k)
                    for d in alt_range[k + 2:max_k]
                ]

                def _intfun(x: float, *args: Tuple[int, int]) -> float:
                    """Helper function for root finding specific to n == max_n case."""
                    return (
                        cdist(args,
                              # Lower bound PDF for alternative
                              (max_k, max_n))[0].pdf(x)
                        - cdist((k, n),
                                # Upper bound PDF for observed
                                (max_k, max_n))[1].pdf(x)
                    )
                # Find intersections through root finding
                intersections += [
                    root_scalar(
                        _intfun,
                        args=tuple(d),
                        bracket=[1e-12, 1 - 1e-12]
                    ).root for d in alt_range[
                        max_k + (1 if (k == max_k - 1) else 0):]
                ]
                intersections = np.array(intersections)
        # Call the parent class constructor.
        super().__init__([k, n],
                         [max_k, max_n],
                         support=support,
                         alt_range=alt_range,
                         cdist=cdist,
                         intersections=intersections,
                         precision=precision,
                         max_poss_bounds=max_poss_bounds)
