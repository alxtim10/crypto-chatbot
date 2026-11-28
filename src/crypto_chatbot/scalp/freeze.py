"""SC-14: Parameter freeze hash for `scalp_v1` (SCALP-SPEC-v1 §12 SC-14).

Single shared SHA-256 function used by BOTH `freeze-params` and `alerts run`:
covers spec params, universe, cost config, and `SCALP_ENGINE_VERSION`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from crypto_chatbot.analysis.forward_proof import compute_params_sha256
from crypto_chatbot.scalp.config import ScalpConfig

SCALP_ENGINE_VERSION = "scalp-v1.0"


def compute_scalp_params_bundle(
    *,
    scalp_config: ScalpConfig,
    universe: Sequence[str],
    engine_version: str = SCALP_ENGINE_VERSION,
) -> tuple[str, dict[str, Any]]:
    """Compute canonical `params_sha256` and component breakdown for `scalp_v1`."""
    spec_dict = scalp_config.model_dump(mode="json")
    # Personal account sizing in `.env` is not a strategy parameter
    spec_dict.pop("position", None)
    cost_dict = scalp_config.costs.model_dump(mode="json")
    sorted_universe = list(universe)

    spec_sha = compute_params_sha256(spec_dict)
    cost_sha = compute_params_sha256(cost_dict)
    universe_sha = compute_params_sha256({"universe": sorted_universe})

    canonical_payload: dict[str, Any] = {
        "strategy": "scalp_v1",
        "engine_version": engine_version,
        "spec_params_sha256": spec_sha,
        "cost_config_sha256": cost_sha,
        "universe_sha256": universe_sha,
    }
    digest = compute_params_sha256(canonical_payload)
    components: dict[str, Any] = {
        "strategy": "scalp_v1",
        "engine_version": engine_version,
        "spec_params": spec_dict,
        "spec_params_sha256": spec_sha,
        "cost_config_sha256": cost_sha,
        "universe_sha256": universe_sha,
        "universe": sorted_universe,
    }
    return digest, components
