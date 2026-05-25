# Processing

Classes for PCA, missing-data PCA with local-ancestry masking, and multi-array ancestry-specific MDS. All three accept `SNPObject` inputs directly and expose `X_new_` after fitting. Embedding tables can be exported with the helper functions below.

```{eval-rst}
.. autoclass:: snputils.processing.pca.PCA
   :members:
```

```{eval-rst}
.. autoclass:: snputils.processing.pca.TorchPCA
   :members:
```

```{eval-rst}
.. autoclass:: snputils.processing.mdpca.mdPCA
   :members:
```

```{eval-rst}
.. autoclass:: snputils.processing.maasmds.maasMDS
   :members:
```

## Embedding Utilities

```{eval-rst}
.. autofunction:: snputils.processing.build_embedding_dataframe
```

```{eval-rst}
.. autofunction:: snputils.processing.embedding_dataframe_from_model
```

```{eval-rst}
.. autofunction:: snputils.processing.save_embedding_table
```

```{eval-rst}
.. autofunction:: snputils.processing.save_embedding_table_from_model
```

```{eval-rst}
.. autofunction:: snputils.processing.try_save_embedding_table
```

```{eval-rst}
.. autofunction:: snputils.processing.embedding_column_names
```

```{eval-rst}
.. autofunction:: snputils.processing.pca_row_haplotype_ids
```
