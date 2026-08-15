from turkicocr.recognition_metrics import evaluate_recognition_gate


def _metrics(**overrides):
    metrics = {
        "val_rec_cer": 0.03,
        "val_rec_rare_char_cer": 0.04,
        "val_rec_chrf": 0.95,
        "val_rec_exact_match_short": 0.95,
        "val_rec_empty_prediction_rate": 0.0,
        "val_rec_too_short_rate": 0.0,
        "val_rec_length_ratio": 1.0,
    }
    metrics.update(overrides)
    return metrics


def test_recognition_gate_passes_strong_result():
    result = evaluate_recognition_gate(_metrics())

    assert result["status"] == "PASS"
    assert result["promote_checkpoint"]


def test_recognition_gate_warns_between_strong_and_fail_thresholds():
    result = evaluate_recognition_gate(_metrics(val_rec_cer=0.07))

    assert result["status"] == "WARN"
    assert result["promote_checkpoint"]


def test_recognition_gate_fails_above_max_cer():
    result = evaluate_recognition_gate(_metrics(val_rec_cer=0.11))

    assert result["status"] == "FAIL"
    assert not result["promote_checkpoint"]


def test_recognition_gate_smoke_fails_above_smoke_threshold():
    result = evaluate_recognition_gate(_metrics(val_rec_cer=0.30))

    assert result["status"] == "SMOKE_FAIL"
    assert not result["promote_checkpoint"]


def test_recognition_gate_does_not_promote_without_val_rec_cer():
    result = evaluate_recognition_gate({"train_loss": 0.1})

    assert result["status"] == "FAIL"
    assert not result["promote_checkpoint"]
    assert "missing val_rec_cer" in result["reasons"]
