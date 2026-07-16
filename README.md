# jax-bGEV

Minimal JAX implementation of sampling and log-probability operations for the GEV and blended GEV distributions described by Castro-Camilo, Huser, and Rue (https://arxiv.org/abs/2106.13110).

The public API is intentionally functional and kwargs-based:

```python
import jax
from jax_bgev import bgev_log_prob, bgev_quantile, bgev_sample, gev_log_prob, gev_sample

key = jax.random.key(0)
x = bgev_sample(key, q=0.0, s=1.0, xi=0.2, shape=(1024,))
logp = bgev_log_prob(x, q=0.0, s=1.0, xi=0.2)
quantiles = bgev_quantile(jax.numpy.array([0.1, 0.5, 0.9]), q=0.0, s=1.0, xi=0.2)
```

Runtime dependencies are kept to JAX only. Inputs are passed through `jax.numpy.asarray` for JAX-native array handling; the package does not depend on NumPy directly.

## Notes

* `q` is the fixed `alpha` quantile and `s` is the `beta` quantile range.
* `gev_*` supports `xi == 0` as the Gumbel limit.
* `bgev_*` requires `xi > 0`, matching the paper's construction that blends a positive-shape Fréchet/GEV right tail with a Gumbel left tail.
* Static scalar parameter validation raises `ValueError`; distribution support violations in log-density calculations return `-inf`.


@misc{castrocamilo2022,
      title={Practical strategies for GEV-based regression models for extremes}, 
      author={Daniela Castro-Camilo and Raphaël Huser and Håvard Rue},
      year={2022},
      eprint={2106.13110},
      archivePrefix={arXiv},
      primaryClass={stat.AP},
      url={https://arxiv.org/abs/2106.13110}, 
}
