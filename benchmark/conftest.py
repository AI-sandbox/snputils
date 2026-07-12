import pytest


def pytest_addoption(parser):
    """
    pytest hook to add custom command line options.
    Must be named pytest_addoption for pytest to recognize it.
    """
    parser.addoption(
        "--memory-profile",
        action="store_true",
        default=False,
        help="Enable memory profiling (slower)"
    )
    parser.addoption(
        "--path",
        action="store",
        help="Path to SNP data"
    )
    parser.addoption(
        "--reader-name",
        action="store",
        default=None,
        help="Run only the benchmark case whose reader name matches this value"
    )
    parser.addoption(
        "--genotype-mode",
        action="store",
        default="dosage",
        choices=("dosage", "phased"),
        help="Whether readers should return genotype dosages or phased allele calls."
    )


@pytest.fixture
def path(request):
    """Fixture to get data path"""
    path = request.config.getoption("--path")
    if not path:
        pytest.skip("No path provided")
    return path


@pytest.fixture
def memory_profile(request):
    """Fixture to check if memory profiling is enabled"""
    return request.config.getoption("--memory-profile")


@pytest.fixture
def reader_name(request):
    """Fixture to get the optional reader-name filter."""
    return request.config.getoption("--reader-name")


@pytest.fixture
def genotype_mode(request):
    """Fixture selecting dosage or phased genotype output."""
    return request.config.getoption("--genotype-mode")
