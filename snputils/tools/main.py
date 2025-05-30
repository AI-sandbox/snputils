from importlib.metadata import version
import sys
import logging

log = logging.getLogger(__name__)


def main(): 
    #snputils_version = version("snputils")
    #log.info(f"snputils - Version {snputils_version}")
    arg_list = tuple(sys.argv)
    assert len(arg_list) > 1, 'Please provide an argument for a tool {"pca", "dummy_tool", "admixture_mapping", "simulation"}.'
    if sys.argv[1] == 'pca':
        from . import pca
        sys.exit(pca.plot_and_save_pca(arg_list[2:]))
    if sys.argv[1] == 'dummy_tool':
        from . import dummy_tool
        sys.exit(dummy_tool.dummy_tool(arg_list[2:]))
    if sys.argv[1] == 'admixture_mapping':
        from . import admixture_mapping
        sys.exit(admixture_mapping.admixmap(arg_list[2:]))
    elif sys.argv[1] == 'simulation':
        from ..simulation import simulator_cli
        sys.exit(simulator_cli.main(arg_list[2:]))
    log.error(f'Invalid argument {arg_list[1]}. Tools available are "pca", "dummy_tool", "admixture_mapping", and "simulation". Please follow the example: "tools_testing.py pca"')
    sys.exit(1)
