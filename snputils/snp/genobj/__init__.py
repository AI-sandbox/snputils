from .snpobj import SNPObject

__all__ = ['SNPObject', 'GRGObject']


def __getattr__(name):
    if name == "GRGObject":
        try:
            from .grgobj import GRGObject
        except ModuleNotFoundError as exc:
            if exc.name == "pygrgl":
                raise ImportError(
                    "GRG support requires the optional dependency 'pygrgl'. "
                    "Install pygrgl separately: "
                    "https://github.com/aprilweilab/grgl#installing-from-pip"
                ) from exc
            raise
        return GRGObject
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
