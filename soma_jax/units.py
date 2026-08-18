"""Unit system for SOMA-JAX body models.

Upstream: ``soma/units.py``
    Faithful port of that code. Unit enum and metres-per-unit conversion.
"""
from __future__ import annotations
from enum import Enum


class Unit(Enum):
    METERS = 1.0
    CENTIMETERS = 0.01
    MILLIMETERS = 0.001

    @property
    def meters_per_unit(self) -> float:
        return self.value

    @property
    def unit_name(self) -> str:
        return self.name.lower()

    @classmethod
    def from_name(cls, name: str) -> Unit:
        name_upper = name.strip().upper()
        for member in cls:
            if member.name == name_upper or member.unit_name == name.strip().lower():
                return member
        aliases = {"m": cls.METERS, "cm": cls.CENTIMETERS, "mm": cls.MILLIMETERS}
        if name.strip().lower() in aliases:
            return aliases[name.strip().lower()]
        raise ValueError(f"Unknown unit: {name!r}. Valid: meters, centimeters, millimeters")

    def to(self, target: Unit) -> float:
        """Scale factor to convert from self to target."""
        return self.meters_per_unit / target.meters_per_unit
