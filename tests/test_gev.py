import jax
import jax.numpy as jnp
import pytest

from jax_bgev import bgev_cdf, bgev_log_prob, bgev_quantile, bgev_sample, gev_cdf, gev_log_prob, gev_quantile, gev_sample, new_to_old


def test_new_to_old_matches_reference_values():
    mu, sigma, xi = new_to_old(q=0.0, s=1.0, xi=0.2)
    assert jnp.isclose(xi, 0.2)
    assert sigma > 0
    assert jnp.isfinite(mu)


def test_gev_quantile_inverts_cdf():
    p = jnp.array([0.1, 0.5, 0.9])
    x = gev_quantile(p, q=1.0, s=2.0, xi=0.2)
    assert jnp.allclose(gev_cdf(x, q=1.0, s=2.0, xi=0.2), p, atol=1e-6)


def test_gev_log_prob_outside_support_is_neginf():
    mu, sigma, xi = new_to_old(q=0.0, s=1.0, xi=0.2)
    lower = mu - sigma / xi
    assert jnp.isneginf(gev_log_prob(lower - 1.0, q=0.0, s=1.0, xi=0.2))


def test_samples_are_jax_arrays_and_finite():
    key = jax.random.key(0)
    x = gev_sample(key, q=0.0, s=1.0, xi=0.2, shape=(16,))
    assert isinstance(x, jax.Array)
    assert x.shape == (16,)
    assert jnp.all(jnp.isfinite(x))


def test_bgev_log_prob_is_finite_in_representable_float32_left_tail():
    x = jnp.array([-10.0, 0.0, 10.0])
    logp = bgev_log_prob(x, q=0.0, s=1.0, xi=0.2)
    assert jnp.all(jnp.isfinite(logp))


def test_gev_gumbel_branch_has_finite_x_gradient():
    grad = jax.grad(lambda x: gev_log_prob(x, q=0.0, s=1.0, xi=0.0))(0.0)
    assert jnp.isfinite(grad)


def test_bgev_cdf_is_monotone_and_quantile_inverts():
    xs = jnp.linspace(-10.0, 10.0, 100)
    cdf = bgev_cdf(xs, q=0.0, s=1.0, xi=0.2)
    assert jnp.all(cdf[1:] >= cdf[:-1])

    p = jnp.array([0.01, 0.1, 0.5, 0.9])
    x = bgev_quantile(p, q=0.0, s=1.0, xi=0.2)
    assert jnp.allclose(bgev_cdf(x, q=0.0, s=1.0, xi=0.2), p, atol=1e-5)


def test_bgev_sample_delegates_to_quantile():
    key = jax.random.key(1)
    x = bgev_sample(key, q=0.0, s=1.0, xi=0.2, shape=(8,))
    assert x.shape == (8,)
    assert jnp.all(jnp.isfinite(x))


def test_static_validation_errors():
    with pytest.raises(ValueError, match="s must be positive"):
        gev_log_prob(0.0, q=0.0, s=-1.0, xi=0.2)
    with pytest.raises(ValueError, match="bGEV requires positive xi"):
        bgev_log_prob(0.0, q=0.0, s=1.0, xi=0.0)
