"""Run the example star. Output goes to ./results_example/KIC1872517/."""
import logging
from pathlib import Path

from numax_proxies import NumaxProxies

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

proxy = NumaxProxies.read_yaml(Path(__file__).parent / "KIC1872517.yaml").run()
proxy.compute_numax_from_acf()
proxy.compute_numax_from_CoV()
proxy.plotting()
print(proxy.results)
