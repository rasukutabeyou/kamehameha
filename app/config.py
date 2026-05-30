from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    max_hp: int = 100
    damage: int = 6
    hit_cooldown_s: float = 0.38
    beam_charge_s: float = 1.15
    beam_duration_s: float = 2.4
    energy_initial: float = 50.0
    energy_max: float = 100.0
    beam_energy_cost: float = 25.0
    energy_charge_per_s: float = 5.0
    energy_charge_damage_bonus: int = 2
    ultra_energy_cost: float = 75.0
    ultra_damage_multiplier: float = 2.0
    ultra_energy_charge_multiplier: float = 2.0
    ultra_damage_reduction: int = 1
    ultra_energy_drain_per_s: float = 3.0

    @classmethod
    def from_env(cls) -> "GameConfig":
        return cls(
            max_hp=int(os.getenv("BEAM_MAX_HP", str(cls.max_hp))),
            damage=int(os.getenv("BEAM_DAMAGE", str(cls.damage))),
            hit_cooldown_s=float(os.getenv("BEAM_HIT_COOLDOWN", str(cls.hit_cooldown_s))),
            beam_charge_s=float(os.getenv("BEAM_CHARGE", str(cls.beam_charge_s))),
            beam_duration_s=float(os.getenv("BEAM_DURATION", str(cls.beam_duration_s))),
            energy_initial=float(os.getenv("BEAM_ENERGY_INITIAL", str(cls.energy_initial))),
            energy_max=float(os.getenv("BEAM_ENERGY_MAX", str(cls.energy_max))),
            beam_energy_cost=float(os.getenv("BEAM_ENERGY_COST", str(cls.beam_energy_cost))),
            energy_charge_per_s=float(os.getenv("BEAM_ENERGY_CHARGE_PER_SEC", str(cls.energy_charge_per_s))),
            energy_charge_damage_bonus=int(os.getenv("BEAM_ENERGY_CHARGE_DAMAGE_BONUS", str(cls.energy_charge_damage_bonus))),
            ultra_energy_cost=float(os.getenv("BEAM_ULTRA_ENERGY_COST", str(cls.ultra_energy_cost))),
            ultra_damage_multiplier=float(os.getenv("BEAM_ULTRA_DAMAGE_MULTIPLIER", str(cls.ultra_damage_multiplier))),
            ultra_energy_charge_multiplier=float(os.getenv("BEAM_ULTRA_ENERGY_CHARGE_MULTIPLIER", str(cls.ultra_energy_charge_multiplier))),
            ultra_damage_reduction=int(os.getenv("BEAM_ULTRA_DAMAGE_REDUCTION", str(cls.ultra_damage_reduction))),
            ultra_energy_drain_per_s=float(os.getenv("BEAM_ULTRA_ENERGY_DRAIN_PER_SEC", str(cls.ultra_energy_drain_per_s))),
        )
