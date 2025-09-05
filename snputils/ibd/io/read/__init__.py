from .functional import read_ibd
from .hap_ibd import HapIBDReader
from .anc_ibd import AncIBDReader
from .auto import IBDReader

__all__ = ['read_ibd', 'HapIBDReader', 'AncIBDReader', 'IBDReader']


