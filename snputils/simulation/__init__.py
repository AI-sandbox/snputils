from importlib import import_module

__all__ = ["OnlineSimulator"]


def __getattr__(name):
    if name == "OnlineSimulator":
        value = getattr(import_module(".simulator", package=__name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
