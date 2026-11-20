"""numax_proxies: numax of solar-like oscillators from the collapsed ACF and the CoV."""
import logging

from .config import GlobalConfig, config_from_dict, load_config
from .pipeline import NumaxProxies

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = "0.5.0"
__all__ = ["NumaxProxies", "load_config", "config_from_dict", "GlobalConfig"]
