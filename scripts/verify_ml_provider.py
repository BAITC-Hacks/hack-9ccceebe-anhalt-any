"""Verify frozen artifact + holdout, raw provider kW and original MODE A metrics. No fit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml import provider
from src.ml.evaluation import evaluate_model
from src.ml.model import scores

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SHA256 = "e2e996fb4289384dd2b776270ce010dfdf1076bb7bf0ea94b19819e88c6e8618"
HOLDOUT_SHA256 = "0d25be3cf926332c2f923f16f5c6c7a6b008b105be74634c6cebafa303e51701"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify():
    artifact, holdout = ROOT / "models/power_model.joblib", ROOT / "data/holdout.csv"
    # CSV checkout may use LF or CRLF. Hash canonical CRLF form used at publication.
    canonical_csv = holdout.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    holdout_digest = hashlib.sha256(canonical_csv).hexdigest()
    if sha256(artifact) != ARTIFACT_SHA256 or holdout_digest != HOLDOUT_SHA256:
        raise ValueError("Published artifact/holdout changed; verification must not retrain or alter holdout")
    model = provider.load_model(artifact)
    frame = pd.read_csv(holdout)
    result = evaluate_model(model, frame, mode="A")
    for form in ["raw", "served"]:
        for metric, expected in model.metadata["metrics"]["test"][form].items():
            if expected is not None:
                np.testing.assert_allclose(result[form][metric], expected, rtol=1e-10, atol=1e-12)
    power = np.empty(len(frame), dtype=float)
    for turbine, group in frame.groupby("turbine_id", sort=False):
        features = group[["wind_speed", "temperature", "wind_direction"]].rename(columns={
            "wind_speed": "wind_speed_ms", "temperature": "temperature_c", "wind_direction": "wind_direction_deg"})
        features.index = pd.DatetimeIndex(pd.to_datetime(group.timestamp, utc=True))
        features.attrs = {"turbine_id": turbine, "wind_height_m": provider.HUB_HEIGHTS_M[turbine]}
        power[group.index] = provider.predict(model, features)
    normalized = scores(frame.power / frame.rated_power, power / frame.rated_power, normalized=True)
    for metric, value in normalized.items():
        np.testing.assert_allclose(value, result["raw"][metric], rtol=1e-10, atol=1e-12)
    return {"mode": "A", "rows": len(frame), "no_retraining": True, "no_holdout_changes": True,
            "artifact_sha256": sha256(artifact), "holdout_sha256_crlf": holdout_digest,
            "training_cutoff": model.metadata["training_end"], "selection_cutoff": model.metadata["validation_end"],
            "test_start": model.metadata["test_start"], "test_end": model.metadata["test_end"],
            "dataframe_api": {"raw": result["raw"], "served": result["served"]},
            "provider_raw_kw": scores(frame.power, power), "provider_raw_normalized": normalized,
            "provider_negative_predictions": int((power < 0).sum()),
            "provider_above_rated_predictions": int((power > frame.rated_power).sum()),
            "artifact_training_versions": model.metadata["versions"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="Optional JSON report path")
    args = parser.parse_args()
    result = verify()
    encoded = json.dumps(result, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
