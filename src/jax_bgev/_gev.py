from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ._validation import finite_or_neginf, validate_common

Array = jax.Array


def _ell(probability: Any, xi: Any) -> Array:
    probability = jnp.asarray(probability)
    xi = jnp.asarray(xi)
    return jnp.where(xi == 0, jnp.log(-jnp.log(probability)), (-jnp.log(probability)) ** (-xi))


def new_to_old(*, q: Any, s: Any, xi: Any, alpha: Any = 0.5, beta: Any = 0.5) -> tuple[Array, Array, Array]:
    """Map quantile/spread parameters ``(q, s, xi)`` to usual ``(mu, sigma, xi)``."""
    validate_common(s=s, alpha=alpha, beta=beta)
    q, s, xi = jnp.asarray(q), jnp.asarray(s), jnp.asarray(xi)
    ell1 = _ell(alpha, xi)
    ell2 = _ell(beta / 2, xi)
    ell3 = _ell(1 - beta / 2, xi)
    sigma0 = s / (ell2 - ell3)
    mu0 = q + sigma0 * ell1
    sigma = xi * s / (ell3 - ell2)
    mu = q - s * (ell1 - 1) / (ell3 - ell2)
    return jnp.where(xi == 0, mu0, mu), jnp.where(xi == 0, sigma0, sigma), xi


def gev_cdf(x: Any, *, q: Any, s: Any, xi: Any, alpha: Any = 0.5, beta: Any = 0.5) -> Array:
    mu, sigma, xi = new_to_old(q=q, s=s, xi=xi, alpha=alpha, beta=beta)
    x = jnp.asarray(x)
    z = (x - mu) / sigma
    safe_xi = jnp.where(xi == 0, 1.0, xi)
    t = 1 + safe_xi * z
    cdf_xi = jnp.exp(-(jnp.maximum(t, 0)) ** (-1 / safe_xi))
    cdf_0 = jnp.exp(-jnp.exp(-z))
    return jnp.where(xi == 0, cdf_0, cdf_xi)


def gev_log_prob(x: Any, *, q: Any, s: Any, xi: Any, alpha: Any = 0.5, beta: Any = 0.5) -> Array:
    mu, sigma, xi = new_to_old(q=q, s=s, xi=xi, alpha=alpha, beta=beta)
    x = jnp.asarray(x)
    z = (x - mu) / sigma
    safe_xi = jnp.where(xi == 0, 1.0, xi)
    t = 1 + safe_xi * z
    logp_0 = -jnp.exp(-z) - jnp.log(sigma) - z
    logp_xi = -(jnp.maximum(t, 0)) ** (-1 / safe_xi) - jnp.log(sigma) - (1 / safe_xi + 1) * jnp.log(t)
    return finite_or_neginf(jnp.where(xi == 0, logp_0, logp_xi), (xi == 0) | (t > 0))


def gev_quantile(p: Any, *, q: Any, s: Any, xi: Any, alpha: Any = 0.5, beta: Any = 0.5) -> Array:
    mu, sigma, xi = new_to_old(q=q, s=s, xi=xi, alpha=alpha, beta=beta)
    p = jnp.asarray(p)
    y = -jnp.log(p)
    q0 = mu - sigma * jnp.log(y)
    safe_xi = jnp.where(xi == 0, 1.0, xi)
    qxi = mu - sigma / safe_xi * (1 - y ** (-safe_xi))
    return jnp.where(xi == 0, q0, qxi)


def gev_sample(key: Array, *, q: Any, s: Any, xi: Any, shape: tuple[int, ...] = (), alpha: Any = 0.5, beta: Any = 0.5) -> Array:
    u = jax.random.uniform(key, shape=shape, minval=jnp.finfo(jnp.float32).tiny, maxval=1.0)
    return gev_quantile(u, q=q, s=s, xi=xi, alpha=alpha, beta=beta)
