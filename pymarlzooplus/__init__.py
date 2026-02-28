def __getattr__(name):
    """
    Lazy import to avoid RuntimeWarning when using 'python3 -m pymarlzooplus.main'.
    This prevents the main module from being loaded before runpy.py executes it.
    """
    if name == 'pymarlzooplus':
        from pymarlzooplus.main import pymarlzooplus
        return pymarlzooplus
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
