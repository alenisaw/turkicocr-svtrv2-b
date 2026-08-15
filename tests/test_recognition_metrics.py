import math

from turkicocr.recognition_metrics import (
    cer,
    chrf_score,
    evaluate_recognition_by_slice,
    evaluate_recognition_predictions,
    exact_match,
    length_ratio,
    rare_char_cer,
    rare_char_confusion_matrix,
    wer,
)


def test_cer_and_wer_known_values():
    assert cer("abc", "abc") == 0.0
    assert cer("abc", "adc") == 1 / 3
    assert wer("бір екі үш", "бір төрт үш") == 1 / 3


def test_chrf_exact_match_and_length_ratio():
    assert chrf_score("Қазақстан", "Қазақстан") == 1.0
    assert 0.0 <= chrf_score("Қазақстан", "Казакстан") <= 1.0
    assert exact_match(" a   b ", "a b")
    assert not exact_match(" a   b ", "a b", normalize=False)
    assert length_ratio("abcd", "ab") == 0.5


def test_rare_char_cer_and_confusion_matrix():
    assert rare_char_cer("қазақ әліпбиі", "қазақ әліпбиі") == 0.0
    assert rare_char_cer("қазақ", "казак") == 1.0
    assert math.isnan(rare_char_cer("москва", "москва"))

    matrix = rare_char_confusion_matrix(["қазақ"], ["казак"])
    assert matrix["қ"]["к"] >= 1


def test_evaluate_recognition_predictions_aggregate_keys():
    metrics = evaluate_recognition_predictions(
        ["Қазақстан", "abc", "ұзын мәтін"],
        ["Қазақстан", "", "ұзын мәтн"],
        [{"text_length_bucket": "short"}, {"text_length_bucket": "short"}, {}],
    )

    assert metrics["val_rec_count"] == 3.0
    assert metrics["val_rec_cer"] > 0
    assert metrics["val_rec_exact_match"] == 1 / 3
    assert metrics["val_rec_exact_match_short"] == 1 / 3
    assert metrics["val_rec_empty_prediction_rate"] == 1 / 3
    assert "val_rec_latency_ms_per_crop" in metrics


def test_evaluate_recognition_by_slice():
    rows = [
        {
            "reference": "Қазақстан",
            "prediction": "Қазақстан",
            "metadata": {"language": "kazakh"},
        },
        {"reference": "Москва", "prediction": "Масква", "metadata": {"language": "russian"}},
    ]

    sliced = evaluate_recognition_by_slice(rows, "language")

    assert [row["language"] for row in sliced] == ["kazakh", "russian"]
    assert sliced[0]["val_rec_cer"] == 0.0
