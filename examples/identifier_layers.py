"""Save a small direct-identifier demonstration without hardware or plotting."""

import argparse
import json
from pathlib import Path

import numpy as np

from reossi.simulation.identifiers import simulate_identifier_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/identifier-layers")
    )
    parser.add_argument("--curvature", type=float, default=5)
    args = parser.parse_args()
    result = simulate_identifier_map(curvature=args.curvature)
    args.output.mkdir(parents=True, exist_ok=False)
    np.save(args.output / "base.npy", result.base)
    np.save(args.output / "fused.npy", result.fused)
    for displacement, (layer, scaled) in enumerate(
        zip(result.layers, result.scaled_layers)
    ):
        np.save(args.output / f"layer_{displacement}.npy", layer)
        np.save(args.output / f"scaled_{displacement}.npy", scaled)
    (args.output / "identifier-simulation.json").write_text(
        json.dumps(result.config, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"output": str(args.output), "shape": list(result.fused.shape)}, indent=2
        )
    )


if __name__ == "__main__":
    main()
