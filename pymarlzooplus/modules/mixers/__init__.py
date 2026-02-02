from .vdn import VDNMixer
from .qmix import QMixer
from .graph_coarsening import GraphCoarsening

REGISTRY = {
    "vdn": VDNMixer,
    "qmix": QMixer,
    "coarsen": GraphCoarsening,
}
