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
        "--sum-strands",
        action="store",
        default="true",
        choices=("true", "false"),
        help="Whether readers should return summed genotype dosages or separate diploid alleles."
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
def sum_strands(request):
    """Fixture to choose summed dosages or separate allele strands."""
    return request.config.getoption("--sum-strands") == "true"
