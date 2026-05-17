import matplotlib

matplotlib.use("Agg")

import pandas as pd

from snputils.visualization import plot_embedding


def test_plot_embedding_groups_by_metadata_column():
    df = pd.DataFrame(
        {
            "PC1": [0.0, 1.0, 2.0],
            "PC2": [2.0, 1.0, 0.0],
            "population": ["CEU", "YRI", "CEU"],
        }
    )

    ax = plot_embedding(df, hue="population", title="PCA")

    assert ax.get_xlabel() == "PC1"
    assert ax.get_ylabel() == "PC2"
    assert ax.get_title() == "PCA"
    assert ax.get_legend() is not None


def test_plot_embedding_validates_columns():
    df = pd.DataFrame({"PC1": [0.0], "group": ["x"]})

    try:
        plot_embedding(df, hue="group")
    except ValueError as exc:
        assert "y column" in str(exc)
    else:
        raise AssertionError("expected missing coordinate column to raise")
