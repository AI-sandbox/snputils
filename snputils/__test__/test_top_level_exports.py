import snputils as su


def test_top_level_analysis_exports(tmp_path):
    assert su.PCA().backend == "sklearn"
    assert su.mdPCA.__name__ == "mdPCA"
    assert su.maasMDS.__name__ == "maasMDS"
    assert su.allele_freq_stream.__name__ == "allele_freq_stream"

    labels_path = tmp_path / "labels.tsv"
    labels_path.write_text("indID\tlabel\nHG001\tEUR\n", encoding="utf-8")
    labels = su.read_labels(labels_path)
    assert labels.to_dict("records") == [{"indID": "HG001", "label": "EUR"}]
