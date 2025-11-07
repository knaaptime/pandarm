import contextlib
from importlib.metadata import PackageNotFoundError, version

from .network import Network

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("pandarm")
