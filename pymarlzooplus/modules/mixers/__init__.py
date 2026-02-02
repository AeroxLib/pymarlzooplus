from .vdn import VDNMixer
from .qmix import QMixer
from .qattn import QAttnMixer
from .graph_coarsening import GraphCoarsening

REGISTRY = {
    "vdn": VDNMixer,
    "qmix": QMixer,
    "qattn": QAttnMixer,
    "coarsen": GraphCoarsening,
}
