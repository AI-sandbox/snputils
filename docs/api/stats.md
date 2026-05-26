# Statistics

Functions for allele-frequency aggregation and population-genetic tests. Inputs may be a `SNPObject` with sample labels or a precomputed allele-frequency tuple from {func}`~snputils.stats.allele_freq_stream`. Block jackknife standard errors are computed by default on f-statistics. Ancestry masking requires a `LocalAncestryObject` when the `ancestry` argument is set.

```{eval-rst}
.. autofunction:: snputils.stats.allele_freq_stream
```

```{eval-rst}
.. autofunction:: snputils.stats.f2
```

```{eval-rst}
.. autofunction:: snputils.stats.f3
```

```{eval-rst}
.. autofunction:: snputils.stats.f4
```

```{eval-rst}
.. autofunction:: snputils.stats.d_stat
```

```{eval-rst}
.. autofunction:: snputils.stats.f4_ratio
```

```{eval-rst}
.. autofunction:: snputils.stats.fst
```

```{eval-rst}
.. autofunction:: snputils.stats.export_qp
```
