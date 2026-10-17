import pytest

from ml.labels import normalize_video_name, parse_timestamp


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("22 секунда", 22.0),
        ("7 секунда", 7.0),
        ("1минута 8 секунда", 68.0),
        ("1 минута 8 секунд", 68.0),
        ("2 минуты", 120.0),
        ("8.5 секунд", 8.5),
        ("1,5 секунды", 1.5),
        ("1 мин 5 сек", 65.0),
        ("01:08", 68.0),
        ("0:57", 57.0),
        (57, 57.0),
        (12.5, 12.5),
        (None, None),
        ("", None),
    ],
)
def test_parse_timestamp(raw, expected):
    assert parse_timestamp(raw) == expected


@pytest.mark.parametrize("raw", ["где-то в середине", "1 минута 60 секунд", "-3 секунды"])
def test_parse_timestamp_rejects_garbage(raw):
    with pytest.raises(ValueError):
        parse_timestamp(raw)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("X-1", "X-1"),
        ("Х-3", "X-3"),  # Cyrillic Х in the dataset
        ("Х-47.mov", "X-47"),
        (" x-12 ", "X-12"),
    ],
)
def test_normalize_video_name(raw, expected):
    assert normalize_video_name(raw) == expected


def test_load_labels_reads_organizer_format(tmp_path):
    from openpyxl import Workbook

    from ml.labels import Label, load_labels

    wb = Workbook()
    ws = wb.active
    ws.append(["наименование файла", "таймстемп", "нарушение / без нарушения"])
    ws.append(["X-1", "22 секунда", "нарушение"])
    ws.append(["Х-3", None, "без нарушения"])
    ws.append(["Х-47", "1минута 8 секунда", "нарушение"])
    path = tmp_path / "labels.xlsx"
    wb.save(path)

    assert load_labels(path) == [
        Label("X-1", True, 22.0),
        Label("X-3", False, None),
        Label("X-47", True, 68.0),
    ]
