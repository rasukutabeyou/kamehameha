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
    ki_initial: float = 50.0
    ki_max: float = 100.0
    beam_ki_cost: float = 25.0
    ki_charge_per_s: float = 5.0
    ki_charge_damage_bonus: int = 2
    super_ki_cost: float = 75.0
    super_damage_multiplier: float = 2.0
    super_ki_charge_multiplier: float = 2.0
    super_damage_reduction: int = 1
    super_ki_drain_per_s: float = 1.0

    @classmethod
    def from_env(cls) -> "GameConfig":
        return cls(
            max_hp=int(os.getenv("KAME_MAX_HP", str(cls.max_hp))),
            damage=int(os.getenv("KAME_DAMAGE", str(cls.damage))),
            hit_cooldown_s=float(os.getenv("KAME_HIT_COOLDOWN", str(cls.hit_cooldown_s))),
            beam_charge_s=float(os.getenv("KAME_BEAM_CHARGE", str(cls.beam_charge_s))),
            beam_duration_s=float(os.getenv("KAME_BEAM_DURATION", str(cls.beam_duration_s))),
            ki_initial=float(os.getenv("KAME_KI_INITIAL", str(cls.ki_initial))),
            ki_max=float(os.getenv("KAME_KI_MAX", str(cls.ki_max))),
            beam_ki_cost=float(os.getenv("KAME_BEAM_KI_COST", str(cls.beam_ki_cost))),
            ki_charge_per_s=float(os.getenv("KAME_KI_CHARGE_PER_SEC", str(cls.ki_charge_per_s))),
            ki_charge_damage_bonus=int(os.getenv("KAME_KI_CHARGE_DAMAGE_BONUS", str(cls.ki_charge_damage_bonus))),
            super_ki_cost=float(os.getenv("KAME_SUPER_KI_COST", str(cls.super_ki_cost))),
            super_damage_multiplier=float(os.getenv("KAME_SUPER_DAMAGE_MULTIPLIER", str(cls.super_damage_multiplier))),
            super_ki_charge_multiplier=float(os.getenv("KAME_SUPER_KI_CHARGE_MULTIPLIER", str(cls.super_ki_charge_multiplier))),
            super_damage_reduction=int(os.getenv("KAME_SUPER_DAMAGE_REDUCTION", str(cls.super_damage_reduction))),
            super_ki_drain_per_s=float(os.getenv("KAME_SUPER_KI_DRAIN_PER_SEC", str(cls.super_ki_drain_per_s))),
        )
