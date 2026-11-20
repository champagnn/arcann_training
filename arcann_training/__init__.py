from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("arcann_training")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

from .__main__ import main  # noqa: E402

__all__ = ["main", "__version__"]
