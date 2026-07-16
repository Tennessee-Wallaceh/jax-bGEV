from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.scipy.special import betaln, betainc

from ._gev import gev_cdf, gev_log_prob, gev_quantile
from ._validation import finite_or_neginf, validate_bgev

Array = jax.Array


def _beta_weight(x: Array, *, a: Array, b: Array, c1: Any, c2: Any) -> tuple[Array, Array]:
    u = jnp.clip((x - a) / (b - a), 0.0, 1.0)
    p = betainc(c1, c2, u)
    log_pdf_u = (c1 - 1) * jnp.log(u) + (c2 - 1) * jnp.log1p(-u) - betaln(c1, c2)
    dp = jnp.where((u > 0) & (u < 1), jnp.exp(log_pdf_u) / (b - a), 0.0)
    return p, dp


def _gumbel_params(*, a: Array, b: Array, alpha: Any, beta: Any, pa: Any, pb: Any) -> tuple[Array, Array]:
    ell_alpha = jnp.log(-jnp.log(alpha))
    ell_pa = jnp.log(-jnp.log(pa))
    ell_pb = jnp.log(-jnp.log(pb))
    ell_beta_lo = jnp.log(-jnp.log(beta / 2))
    ell_beta_hi = jnp.log(-jnp.log(1 - beta / 2))
    q_tilde = a - (b - a) * (ell_alpha - ell_pa) / (ell_pa - ell_pb)
    s_tilde = (b - a) * (ell_beta_lo - ell_beta_hi) / (ell_pa - ell_pb)
    return q_tilde, s_tilde


def _components(*, q: Any, s: Any, xi: Any, alpha: Any, beta: Any, pa: Any, pb: Any) -> tuple[Array, Array, Array, Array]:
    a = gev_quantile(pa, q=q, s=s, xi=xi, alpha=alpha, beta=beta)
    b = gev_quantile(pb, q=q, s=s, xi=xi, alpha=alpha, beta=beta)
    qg, sg = _gumbel_params(a=a, b=b, alpha=alpha, beta=beta, pa=pa, pb=pb)
    return a, b, qg, sg


def bgev_cdf(x: Any, *, q: Any, s: Any, xi: Any, alpha: Any = 0.5, beta: Any = 0.5, pa: Any = 0.05, pb: Any = 0.2, c1: Any = 5.0, c2: Any = 5.0) -> Array:
    validate_bgev(s=s, xi=xi, alpha=alpha, beta=beta, pa=pa, pb=pb, c1=c1, c2=c2)
    x = jnp.asarray(x)
    a, b, qg, sg = _components(q=q, s=s, xi=xi, alpha=alpha, beta=beta, pa=pa, pb=pb)
    weight, _ = _beta_weight(x, a=a, b=b, c1=c1, c2=c2)
    log_f = jnp.log(jnp.clip(gev_cdf(x, q=q, s=s, xi=xi, alpha=alpha, beta=beta), min=jnp.finfo(x.dtype).tiny))
    log_g = jnp.log(jnp.clip(gev_cdf(x, q=qg, s=sg, xi=0.0, alpha=alpha, beta=beta), min=jnp.finfo(x.dtype).tiny))
    return jnp.exp(weight * log_f + (1 - weight) * log_g)


def bgev_log_prob(x: Any, *, q: Any, s: Any, xi: Any, alpha: Any = 0.5, beta: Any = 0.5, pa: Any = 0.05, pb: Any = 0.2, c1: Any = 5.0, c2: Any = 5.0) -> Array:
    validate_bgev(s=s, xi=xi, alpha=alpha, beta=beta, pa=pa, pb=pb, c1=c1, c2=c2)
    x = jnp.asarray(x)
    tiny = jnp.finfo(x.dtype).tiny
    a, b, qg, sg = _components(q=q, s=s, xi=xi, alpha=alpha, beta=beta, pa=pa, pb=pb)
    weight, dweight = _beta_weight(x, a=a, b=b, c1=c1, c2=c2)
    f_cdf = jnp.clip(gev_cdf(x, q=q, s=s, xi=xi, alpha=alpha, beta=beta), min=tiny)
    g_cdf = jnp.clip(gev_cdf(x, q=qg, s=sg, xi=0.0, alpha=alpha, beta=beta), min=tiny)
    log_f = jnp.log(f_cdf)
    log_g = jnp.log(g_cdf)
    h_cdf = jnp.exp(weight * log_f + (1 - weight) * log_g)
    f_ratio = jnp.exp(gev_log_prob(x, q=q, s=s, xi=xi, alpha=alpha, beta=beta) - log_f)
    g_ratio = jnp.exp(gev_log_prob(x, q=qg, s=sg, xi=0.0, alpha=alpha, beta=beta) - log_g)
    hazard = dweight * (log_f - log_g) + weight * f_ratio + (1 - weight) * g_ratio
    return finite_or_neginf(jnp.log(h_cdf) + jnp.log(hazard), hazard > 0)


def bgev_sample(key: Array, *, q: Any, s: Any, xi: Any, shape: tuple[int, ...] = (), alpha: Any = 0.5, beta: Any = 0.5, pa: Any = 0.05, pb: Any = 0.2, c1: Any = 5.0, c2: Any = 5.0, steps: int = 80) -> Array:
    validate_bgev(s=s, xi=xi, alpha=alpha, beta=beta, pa=pa, pb=pb, c1=c1, c2=c2)
    u = jax.random.uniform(key, shape=shape, minval=jnp.finfo(jnp.float32).tiny, maxval=1.0)
    lo = gev_quantile(jnp.minimum(u, 1e-6), q=q, s=s, xi=0.0, alpha=alpha, beta=beta) - jnp.asarray(s) * 10
    hi = gev_quantile(jnp.maximum(u, 1 - 1e-6), q=q, s=s, xi=xi, alpha=alpha, beta=beta) + jnp.asarray(s) * 10

    def body(_, bounds):
        lower, upper = bounds
        mid = (lower + upper) / 2
        go_right = bgev_cdf(mid, q=q, s=s, xi=xi, alpha=alpha, beta=beta, pa=pa, pb=pb, c1=c1, c2=c2) < u
        return jnp.where(go_right, mid, lower), jnp.where(go_right, upper, mid)

    lo, hi = jax.lax.fori_loop(0, steps, body, (lo, hi))
    return (lo + hi) / 2
