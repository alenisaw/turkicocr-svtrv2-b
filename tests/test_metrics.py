from turkicocr.metrics import cer, chrf, normalize_text
from turkicocr.rare_chars import contains_rare_char, rare_char_stats


def test_cer_exact():
    assert cer("abc", "abc") == 0


def test_cer_nonzero():
    assert cer("abc", "adc") > 0


def test_chrf_range():
    score = chrf("Қазақстан", "Казакстан")
    assert 0 <= score <= 1


def test_normalize_text():
    assert normalize_text(" a   b ") == "a b"


def test_rare_chars():
    assert contains_rare_char("қазақ")
    stats = rare_char_stats(["қазақ"], ["казак"])
    assert stats.gt_count > 0
