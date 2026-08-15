from turkicocr.page_reconstruction import assign_lines_to_zone, reconstruct_zone_from_lines


def test_assign_lines_to_zone_uses_page_and_overlap():
    zone = {"sample_id": "z1", "source_page_id": "p1", "bbox": [0, 0, 100, 100]}
    lines = [
        {"sample_id": "l1", "source_page_id": "p1", "bbox": [10, 10, 90, 20]},
        {"sample_id": "l2", "source_page_id": "p2", "bbox": [10, 10, 90, 20]},
        {"sample_id": "l3", "source_page_id": "p1", "bbox": [200, 10, 250, 20]},
    ]

    assigned = assign_lines_to_zone(zone, lines)

    assert [row["sample_id"] for row in assigned] == ["l1"]


def test_reconstruct_zone_from_lines_sorts_geometrically():
    zone = {"sample_id": "z1", "source_page_id": "p1", "bbox": [0, 0, 100, 100], "text": "one\ntwo"}
    lines = [
        {"sample_id": "l2", "source_page_id": "p1", "bbox": [0, 50, 80, 60]},
        {"sample_id": "l1", "source_page_id": "p1", "bbox": [0, 10, 80, 20]},
    ]

    result = reconstruct_zone_from_lines(zone, lines, {"l1": "one", "l2": "two"})

    assert result["prediction"] == "one\ntwo"
    assert result["line_ids"] == ["l1", "l2"]
