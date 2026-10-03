"""fgcsim: drone ArUco-mapping simulator for AeroClub IITD's Tech FGC problem statement."""
from .config import PARTS, PUBLIC_SALT, get_part
from .env import FieldSim, Observation, decode_jpeg
from .world import generate_world

__all__ = ["FieldSim", "Observation", "decode_jpeg", "generate_world", "get_part", "PARTS", "PUBLIC_SALT"]
__version__ = "1.0.0"
