import json
import sys
import pytest
from scripts.train_model import main
from scripts.evaluate_model import main as evaluate_main


def test_evaluate_missing_model(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["evaluate_model", str(tmp_path / "holdout.csv"),
                                    "--model", str(tmp_path / "missing.joblib"), "--mode", "A"])
    with pytest.raises(SystemExit) as error:
        evaluate_main()
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "Trained model not found" in message
    assert "Running tests does not create" in message


def test_evaluate_missing_holdout_before_loading_model(tmp_path, monkeypatch, capsys):
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"placeholder: must not be loaded before path validation")
    monkeypatch.setattr(sys, "argv", ["evaluate_model", str(tmp_path / "missing.csv"),
                                    "--model", str(artifact), "--mode", "A"])
    with pytest.raises(SystemExit) as error:
        evaluate_main()
    assert error.value.code == 2
    assert "Holdout dataset not found" in capsys.readouterr().err


def test_missing_config_has_actionable_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["train_model", str(tmp_path / "scada.csv"),
                                    "--config", str(tmp_path / "missing.json")])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "Copy examples/config.json" in capsys.readouterr().err


def test_missing_dataset_has_actionable_error(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_model", str(tmp_path / "scada.csv"), "--config", str(config)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "no dataset is bundled" in capsys.readouterr().err


def test_empty_target_source_and_bom_config(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"target_source": None}), encoding="utf-8-sig")
    dataset = tmp_path / "scada.csv"
    dataset.write_text("timestamp,power\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_model", str(dataset), "--config", str(config)])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "Set target_source" in capsys.readouterr().err
