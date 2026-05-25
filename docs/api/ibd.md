# IBD

Identity-by-descent segment objects and readers for hap-IBD and ancIBD outputs. Segments can be filtered and trimmed to ancestry tracts with {meth}`~snputils.IBDObject.restrict_to_ancestry` when local ancestry is available.

## Objects

```{eval-rst}
.. autoclass:: snputils.IBDObject
   :members:
```

## Readers

```{eval-rst}
.. autoclass:: snputils.IBDReader
   :members:
```

```{eval-rst}
.. autoclass:: snputils.HapIBDReader
   :members:
```

```{eval-rst}
.. autoclass:: snputils.AncIBDReader
   :members:
```

## Read Functions

```{eval-rst}
.. autofunction:: snputils.read_ibd
```
