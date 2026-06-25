import snputils as su


def test_top_level_analysis_exports(tmp_path):
    assert su.BCFReader.__name__ == "BCFReader"
    assert su.BCFWriter.__name__ == "BCFWriter"
    assert su.read_bcf.__name__ == "read_bcf"
    assert su.PCA().backend == "sklearn"
    assert su.mdPCA.__name__ == "mdPCA"
    assert su.maasMDS.__name__ == "maasMDS"
    assert su.allele_freq_stream.__name__ == "allele_freq_stream"
    assert su.FLAREReader.__name__ == "FLAREReader"
    assert su.FLAREWriter.__name__ == "FLAREWriter"
    assert su.read_flare.__name__ == "read_flare"

    labels_path = tmp_path / "labels.tsv"
    labels_path.write_text("indID\tlabel\nHG001\tEUR\n", encoding="utf-8")
    labels = su.read_labels(labels_path)
    assert labels.to_dict("records") == [{"indID": "HG001", "label": "EUR"}]
