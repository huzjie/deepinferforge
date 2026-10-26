# -*- coding: utf-8 -*-
"""Backend registry: name -> Backend class."""
_BACKENDS = {}


def register_backend(name):
    def deco(cls):
        _BACKENDS[name] = cls
        cls.backend_name = name
        return cls
    return deco


def get_backend(name):
    if name not in _BACKENDS:
        raise KeyError("unknown backend %r (available: %s)"
                       % (name, ", ".join(sorted(_BACKENDS))))
    return _BACKENDS[name]


def list_backends():
    return sorted(_BACKENDS)


def get_backend_names():
    return sorted(_BACKENDS)
