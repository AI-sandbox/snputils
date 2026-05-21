import numpy as np

from snputils.ancestry.io.wide.read import read_admixture


def test_read_admixture_accepts_dotted_prefix_with_q_and_p(tmp_path):
    prefix = tmp_path / "admix.10.10"
    q = np.array([[0.25, 0.75], [0.6, 0.4]])
    p = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    np.savetxt(f"{prefix}.Q", q, delimiter=" ")
    np.savetxt(f"{prefix}.P", p, delimiter=" ")

    observed = read_admixture(prefix)

    np.testing.assert_allclose(observed.Q, q)
    np.testing.assert_allclose(observed.P, p)
