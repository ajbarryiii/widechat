"""Validate a recorded flash backend smoke artifact.

Example:
python -m scripts.validate_blackwell_smoke_artifact --artifact-json artifacts/flash_backend_smoke.json --require-blackwell --expect-backend fa4
"""

import argparse
import json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate flash backend smoke artifact")
    parser.add_argument("--artifact-json", required=True, help="artifact produced by scripts.flash_backend_smoke --output-json")
    parser.add_argument("--expect-backend", choices=["fa4", "fa3", "sdpa"], default="fa4", help="required selected backend")
    parser.add_argument("--require-blackwell", action="store_true", help="require sm100+ CUDA capability in artifact")
    return parser.parse_args()


def _load_artifact(path: str) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("artifact must be a JSON object")
    return payload


def _parse_capability(raw: object) -> tuple[int, int] | None:
    if raw is None:
        return None
    if not isinstance(raw, list) or len(raw) != 2 or not all(isinstance(v, int) for v in raw):
        raise ValueError("cuda_capability must be [major, minor] or null")
    return raw[0], raw[1]


def _validate_artifact(payload: dict[str, object], expect_backend: str, require_blackwell: bool) -> tuple[str, tuple[int, int] | None]:
    selected_backend = payload.get("selected_backend")
    if not isinstance(selected_backend, str):
        raise ValueError("artifact missing selected_backend string")
    if selected_backend != expect_backend:
        raise RuntimeError(f"expected backend {expect_backend}, got {selected_backend}")

    cuda_available = payload.get("cuda_available")
    if not isinstance(cuda_available, bool):
        raise ValueError("artifact missing cuda_available bool")

    capability = _parse_capability(payload.get("cuda_capability"))
    if require_blackwell:
        if not cuda_available:
            raise RuntimeError("artifact does not report CUDA availability")
        if capability is None:
            raise RuntimeError("artifact missing cuda_capability")
        major, _minor = capability
        if major < 10:
            raise RuntimeError("artifact is not from Blackwell (requires sm100+)")

    return selected_backend, capability


def main() -> None:
    args = _parse_args()
    payload = _load_artifact(args.artifact_json)
    selected_backend, capability = _validate_artifact(payload, args.expect_backend, args.require_blackwell)

    if capability is None:
        capability_str = "none"
    else:
        capability_str = f"sm{capability[0]}{capability[1]}"
    print(f"artifact_ok selected={selected_backend} capability={capability_str}")


if __name__ == "__main__":
    main()
