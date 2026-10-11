from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Plan:
    strategy: str
    k: float
    beta: float
    additive_scale: float
    sites: tuple[int, ...]

    @property
    def enabled(self):
        return self.strategy != "none"


def parse_plan(extra: dict | None, manifest: dict) -> Plan:
    extra = extra or {}
    known = {
        "tact_schema",
        "tact_bundle_id",
        "tact_strategy",
        "tact_k",
        "tact_lambda",
        "tact_additive_scale",
        "tact_site_ids",
    }
    unknown = {key for key in extra if key.startswith("tact_")} - known
    if unknown or not known.issubset(extra):
        raise ValueError(f"Invalid/missing TACT request fields: {unknown or known - extra.keys()}")
    if type(extra["tact_schema"]) is not int or extra["tact_schema"] != 1:
        raise ValueError("Unsupported TACT request schema")
    if extra["tact_bundle_id"] != manifest["bundle_id"]:
        raise ValueError("TACT request bundle identity mismatch")
    strategy = extra["tact_strategy"]
    if strategy not in {"none", "cap", "gated", "add"}:
        raise ValueError("Unsupported TACT strategy")
    vals = []
    for key in ("tact_k", "tact_lambda", "tact_additive_scale"):
        value = extra[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Non-finite/non-numeric TACT parameter: {key}")
        vals.append(float(value))
    k, lam, additive = vals
    if k < 0 or not 0 <= lam <= 1:
        raise ValueError("TACT requires k>=0 and lambda in [0,1]")
    sites = extra["tact_site_ids"]
    if (
        not isinstance(sites, list)
        or any(type(i) is not int for i in sites)
        or len(sites) != len(set(sites))
        or any(i < 0 or i >= len(manifest["sites"]) for i in sites)
    ):
        raise ValueError("TACT requested an invalid/uncompiled site")
    if strategy != "none" and not sites:
        raise ValueError("Enabled TACT plan needs sites")
    return Plan(
        strategy,
        k,
        1.0 if strategy == "cap" else lam if strategy == "gated" else 0.0,
        additive if strategy == "add" else 0.0,
        tuple(sites),
    )
