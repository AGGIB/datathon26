"""Only synthetic fixtures; never actual competition labels."""
import importlib.util
import unittest
import csv
import tempfile
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def synthetic_workbook(path, rows):
    """Minimal real OOXML package containing synthetic input only."""
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    data = ''.join('<row>' + ''.join('<c r="%s%d" t="inlineStr"><is><t>%s</t></is></c>' % (chr(65 + i), r + 1, escape(str(v))) for i, v in enumerate(row)) + '</row>' for r, row in enumerate(rows))
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('xl/workbook.xml', '<workbook xmlns="%s" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Labels" sheetId="1" r:id="rId1"/></sheets></workbook>' % ns)
        archive.writestr('xl/_rels/workbook.xml.rels', '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Target="worksheets/sheet1.xml" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"/></Relationships>')
        archive.writestr('xl/worksheets/sheet1.xml', '<worksheet xmlns="%s"><sheetData>%s</sheetData></worksheet>' % (ns, data))


class TimestampTests(unittest.TestCase):
    def test_huge_numeric_timestamp_is_explicit_value_error(self):
        from ml.evaluation_labels import parse_timestamp
        with self.assertRaises(ValueError):
            parse_timestamp(10 ** 1000)

    def test_supported_timestamps(self):
        self.assertIsNotNone(importlib.util.find_spec('ml.evaluation_labels'), 'timestamp parser is not implemented')
        from ml.evaluation_labels import parse_timestamp
        for value, expected in [(22, 22), ('22 секунда', 22), ('1минута 8 секунда', 68), ('01:08', 68), ('1:02:03', 3723), ('1.5', 1.5)]:
            with self.subTest(value=value):
                self.assertEqual(parse_timestamp(value), expected)
        for value in [-1, float('nan'), float('inf'), '-01:20', '1:60', '1:2:3:4', 'garbage', '22 sec rubbish', True, '']:
            with self.subTest(invalid=value):
                with self.assertRaises(ValueError):
                    parse_timestamp(value)


class LabelTests(unittest.TestCase):
    def test_numeric_xlsx_identifiers_are_not_silently_reformatted(self):
        from ml.evaluation_labels import load_labels
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'numeric_id.xlsx'
            synthetic_workbook(path, [['video', 'timestamp', 'label'], ['dummy', '1', 'нарушение']])
            with zipfile.ZipFile(path) as archive:
                contents = {name: archive.read(name) for name in archive.namelist()}
            member = 'xl/worksheets/sheet1.xml'
            contents[member] = contents[member].replace(b'<c r="A2" t="inlineStr"><is><t>dummy</t></is></c>', b'<c r="A2"><v>123</v></c>')
            with zipfile.ZipFile(path, 'w') as archive:
                for name, content in contents.items():
                    archive.writestr(name, content)
            with self.assertRaises(ValueError):
                load_labels(path)

    def test_xlsx_shared_strings_sparse_and_numeric_seconds(self):
        from ml.evaluation_labels import load_labels
        ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'numeric.xlsx'
            synthetic_workbook(path, [['video', 'timestamp', 'label']])
            with zipfile.ZipFile(path) as source:
                contents = {name: source.read(name) for name in source.namelist()}
            contents['xl/sharedStrings.xml'] = '<sst xmlns="%s"><si><t>X_synthetic</t></si><si><t>нарушение</t></si><si><t>Х_synthetic_n</t></si><si><t>без нарушения</t></si></sst>' % ns
            contents['xl/worksheets/sheet1.xml'] = '<worksheet xmlns="%s"><sheetData><row><c r="A1" t="inlineStr"><is><t>video</t></is></c><c r="B1" t="inlineStr"><is><t>timestamp</t></is></c><c r="C1" t="inlineStr"><is><t>label</t></is></c></row><row><c r="A2" t="s"><v>0</v></c><c r="B2"><v>1E+2</v></c><c r="C2" t="s"><v>1</v></c></row><row><c r="A3" t="s"><v>2</v></c><c r="C3" t="s"><v>3</v></c></row></sheetData></worksheet>' % ns
            with zipfile.ZipFile(path, 'w') as archive:
                for name, content in contents.items():
                    archive.writestr(name, content)
            rows = load_labels(path)
            self.assertEqual(rows[0]['timestamp'], 100)
            self.assertIsNone(rows[1]['timestamp'])
            self.assertEqual(rows[1]['video'], 'Х_synthetic_n')

    def test_csv_and_xlsx_preserve_identifiers_and_truth(self):
        import ml.evaluation_labels as labels
        self.assertTrue(hasattr(labels, 'load_labels'), 'label loader missing')
        rows = [['наименование файла', 'таймстемп', 'нарушение / без нарушения'], ['X_synthetic.mov', '1минута 8 секунда', 'нарушение'], ['Х_synthetic_n', '', 'без нарушения']]
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            with (folder / 'labels.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                csv.writer(stream).writerows(rows)
            synthetic_workbook(folder / 'labels.xlsx', rows)
            csv_labels = labels.load_labels(folder / 'labels.csv')
            self.assertEqual(csv_labels, labels.load_labels(folder / 'labels.xlsx'))
            self.assertEqual([row['video'] for row in csv_labels], ['X_synthetic.mov', 'Х_synthetic_n'])
            self.assertEqual([row['positive'] for row in csv_labels], [True, False])
            self.assertEqual(csv_labels[0]['timestamp'], 68)
            self.assertIsNone(csv_labels[1]['timestamp'])
            self.assertEqual(labels.video_key('X_synthetic.mov'), 'X_SYNTHETIC')
            self.assertEqual(labels.video_key('X_synthetic'), labels.video_key('Х_synthetic'))
            self.assertEqual(labels.video_key('X_synthetic.mp4'), 'X_SYNTHETIC')
            with (folder / 'bad.csv').open('w', encoding='utf-8', newline='') as stream:
                csv.writer(stream).writerows([rows[0], ['same', '', 'без нарушения'], ['same.mov', '', 'нарушение']])
            with self.assertRaises(ValueError):
                labels.load_labels(folder / 'bad.csv')


if __name__ == '__main__':
    unittest.main()
