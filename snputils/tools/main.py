import sys
import logging

log = logging.getLogger(__name__)


def main(): 
    arg_list = tuple(sys.argv)
    assert len(arg_list) > 1, 'Please provide an argument for a tool {"pca", "admixture_mapping", "admixed_simulation"}.'
    if sys.argv[1] == 'pca':
        from . import pca
        sys.exit(pca.plot_and_save_pca(arg_list[2:]))
    elif sys.argv[1] == 'admixture_mapping':
        from . import admixture_simulation
        sys.exit(admixture_simulation.simulate_admixed_individuals(arg_list[2:]))
    elif sys.argv[1] == "admixed_simulation":
        from . import admixture_mapping
        sys.exit(admixture_mapping.admixmap(arg_list[2:]))
    log.error(f'Invalid argument {arg_list[1]}. Tools available are "pca", "admixture_mapping", and "admixed_simulation".')
    sys.exit(1)

if __name__ == '__main__':
    main()
