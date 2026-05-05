# Visualization

Visualization utilities produce figures for global ancestry, local ancestry, dimensionality reduction, and admixture mapping results.

```python
from snputils.visualization import scatter

fig = scatter(x, y, labels=labels)
```

Local ancestry visualization supports chromosome-painting style views and dataset-level summaries. See the local ancestry tutorial for an end-to-end example with rendered figures.

```{image} ../../assets/lai_dataset_level.png
:alt: Dataset-level local ancestry visualization
:width: 760px
```
