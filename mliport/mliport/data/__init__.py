"""Packaged read-only data resources shipped inside mliport wheels.

This package exists so ``importlib.resources.files("mliport.data")`` has real
Package semantics on every supported Python (3.10-3.12). The curated
capability registry lives in ``mliport/data/validation/`` and is consumed by
:mod:`mliport.capabilities`.
"""
