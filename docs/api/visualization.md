# Visualization

Plotting helpers for embeddings, local ancestry, global admixture proportions, and association summaries. Most functions accept in-memory objects or result files with the column names documented in each signature. Scatter plots expect a fitted PCA, mdPCA, or maasMDS model with `X_new_` and `samples_`.

```{eval-rst}
.. autofunction:: snputils.visualization.scatter
```

```{eval-rst}
.. autofunction:: snputils.visualization.lai.plot_lai
```

```{eval-rst}
.. autofunction:: snputils.visualization.admixture.reorder_admixture
```

```{eval-rst}
.. autofunction:: snputils.visualization.admixture.plot_admixture
```

```{eval-rst}
.. autofunction:: snputils.visualization.manhattan_plot.manhattan_plot
```

```{eval-rst}
.. autofunction:: snputils.visualization.qq_plot.qq_plot
```

```{eval-rst}
.. autofunction:: snputils.visualization.admixture_viz.pong_viz
```

```{eval-rst}
.. autofunction:: snputils.visualization.admixture_viz.create_filemap
```

```{eval-rst}
.. autofunction:: snputils.visualization.chromosome_painting
```
