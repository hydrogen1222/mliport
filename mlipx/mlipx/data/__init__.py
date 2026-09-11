"""Packaged read-only data resources shipped inside mlipx wheels.

This package exists so ``importlib.resources.files("mlipx.data")`` has real
Package semantics on every supported Python (3.10-3.12). The curated
capability registry lives in ``mlipx/data/validation/`` and is consumed by
:mod:`mlipx.capabilities`.
"""
