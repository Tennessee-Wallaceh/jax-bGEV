from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def _all_static_positive(name: str, value: Any) -> None:
    if isinstance(value, (int, float)) and not value > 0:
        raise ValueError(f"{name} must be positive; got {value!r}.")


def _all_static_probability(name: str, value: Any) -> None:
    if isinstance(value, (int, float)) and not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1; got {value!r}.")


def validate_common(*, s: Any, alpha: Any, beta: Any) -> None:
    _all_static_positive("s", s)
    _all_static_probability("alpha", alpha)
    _all_static_probability("beta", beta)


def validate_bgev(*, s: Any, xi: Any, alpha: Any, beta: Any, pa: Any, pb: Any, c1: Any, c2: Any) -> None:
    validate_common(s=s, alpha=alpha, beta=beta)
    _all_static_positive("xi", xi)
    _all_static_probability("pa", pa)
    _all_static_probability("pb", pb)
    _all_static_positive("c1", c1)
    _all_static_positive("c2", c2)
    if isinstance(pa, (int, float)) and isinstance(pb, (int, float)) and not pa < pb:
        raise ValueError(f"pa must be less than pb; got pa={pa!r}, pb={pb!r}.")
    if isinstance(c1, (int, float)) and c1 <= 3:
        raise ValueError(f"c1 must be greater than 3 for the paper's smoothness condition; got {c1!r}.")
    if isinstance(c2, (int, float)) and c2 <= 3:
        raise ValueError(f"c2 must be greater than 3 for the paper's smoothness condition; got {c2!r}.")
    if isinstance(xi, (int, float)) and xi <= 0:
        raise ValueError(f"bGEV requires positive xi; use GEV/Gumbel operations for xi <= 0; got {xi!r}.")


def finite_or_neginf(log_density: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(valid, log_density, -jnp.inf)
