from turkicocr.page_reconstruction import reconstruct_page_text, sort_zones_geometric


def test_sort_zones_geometric_top_to_bottom_left_to_right():
    zones = [
        {"zone_id": "b", "bbox": [100, 10, 150, 30]},
        {"zone_id": "c", "bbox": [10, 50, 90, 70]},
        {"zone_id": "a", "bbox": [10, 10, 90, 30]},
    ]

    ordered = sort_zones_geometric(zones)

    assert [zone["zone_id"] for zone in ordered] == ["a", "b", "c"]


def test_reconstruct_page_text_from_prediction_mapping():
    zones = [
        {"zone_id": "second", "bbox": [10, 50, 90, 70]},
        {"zone_id": "first", "bbox": [10, 10, 90, 30]},
    ]
    predictions = {"first": "Қазақстан Республикасы", "second": "Алматы қаласы"}

    text = reconstruct_page_text(zones, predictions)

    assert text == "Қазақстан Республикасы\nАлматы қаласы"
