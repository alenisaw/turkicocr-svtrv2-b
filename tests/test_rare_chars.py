from turkicocr.rare_chars import RARE_CHARS, contains_rare_char
from turkicocr.recognition_metrics import rare_char_cer, rare_char_confusion_matrix


def test_turkic_rare_chars_list():
    assert RARE_CHARS == ("ә", "ғ", "қ", "ң", "ө", "ұ", "ү", "і", "һ")


def test_detects_rare_chars_in_text():
    assert contains_rare_char("Қазақ әліпбиі")
    assert not contains_rare_char("Русский текст")


def test_rare_char_cer_tracks_collapse():
    assert rare_char_cer("ә ғ қ ң ө ұ ү і һ", "а г к н о у у и х") == 1.0


def test_rare_char_confusion_tracking():
    matrix = rare_char_confusion_matrix(["әліпби қ"], ["аліпби к"])
    assert matrix["ә"]["а"] == 1
    assert matrix["қ"]["к"] == 1
