"""Strict local-only label readers; no third-party packages required."""
import math
import re
import csv
import posixpath
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


def video_key(value):
    """Only an exact optional lowercase .mov suffix is equivalent."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError('video identifier must be a nonempty string without edge whitespace')
    if '/' in value or '\\' in value:
        raise ValueError('video identifier must not contain a path')
    return value[:-4] if value.endswith('.mov') else value


def read_table(path):
    """Read UTF-8 CSV (comma/semicolon) or first XLSX worksheet, preserving cells."""
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            sample = stream.read(4096)
            stream.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
            except csv.Error:
                dialect = csv.excel
            return list(csv.reader(stream, dialect))
    if path.suffix.lower() != '.xlsx':
        raise ValueError('labels must be CSV or XLSX')
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared = [''.join(node.itertext()) for node in ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', ns)]
        workbook = ET.fromstring(archive.read('xl/workbook.xml'))
        sheet = workbook.find('m:sheets/m:sheet', ns)
        if sheet is None:
            raise ValueError('workbook has no sheets')
        relation_id = sheet.attrib.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
        relations = ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        target = next((node.attrib['Target'] for node in relations if node.attrib.get('Id') == relation_id), None)
        if not target:
            raise ValueError('worksheet relationship missing')
        member = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/' + target)
        root = ET.fromstring(archive.read(member))
        rows = []
        for row in root.findall('m:sheetData/m:row', ns):
            values = []
            for cell in row.findall('m:c', ns):
                letters = re.match(r'[A-Z]+', cell.attrib.get('r', ''))
                index = 0
                if letters:
                    for letter in letters.group():
                        index = index * 26 + ord(letter) - 64
                    index -= 1
                else:
                    index = len(values)
                values.extend([''] * (index + 1 - len(values)))
                if cell.find('m:f', ns) is not None:
                    raise ValueError('formula cells are not accepted in labels')
                value = cell.find('m:v', ns)
                text = value.text if value is not None else ''
                kind = cell.attrib.get('t')
                if kind == 's':
                    text = shared[int(text)]
                elif kind == 'inlineStr':
                    text = ''.join(cell.find('m:is', ns).itertext())
                elif kind in ('b', 'e'):
                    raise ValueError('boolean/error label cells are unsupported')
                elif kind in (None, 'n') and text:
                    text = float(text)
                values[index] = text if text is not None else ''
            rows.append(values)
        return rows


def load_labels(path):
    """Return video, positive and optional timestamp; reject ambiguous labels."""
    rows = read_table(path)
    if not rows:
        raise ValueError('empty label table')
    aliases = {'video': ('video', 'наименование файла'), 'timestamp': ('timestamp', 'таймстемп'), 'label': ('label', 'нарушение / без нарушения')}
    header = [str(cell).strip().lower() for cell in rows[0]]
    columns = {}
    for name, choices in aliases.items():
        found = [i for i, cell in enumerate(header) if cell in choices]
        if len(found) != 1:
            raise ValueError('missing or duplicate label column: ' + name)
        columns[name] = found[0]
    result, seen = [], set()
    for number, row in enumerate(rows[1:], 2):
        if not any(str(cell).strip() for cell in row):
            continue
        def cell(name):
            index = columns[name]
            return row[index] if index < len(row) else ''
        video = cell('video')
        if not isinstance(video, str):
            raise ValueError('video identifier cells must be text, not numeric')
        key = video_key(video)
        if key in seen:
            raise ValueError('duplicate video identifier at row %d' % number)
        seen.add(key)
        label = str(cell('label')).strip().lower()
        if label not in ('нарушение', 'без нарушения'):
            raise ValueError('unknown binary label at row %d' % number)
        timestamp = cell('timestamp')
        timestamp = None if timestamp is None or str(timestamp).strip() == '' else parse_timestamp(timestamp)
        result.append({'video': video, 'positive': label == 'нарушение', 'timestamp': timestamp})
    if not result:
        raise ValueError('no label records')
    return result


def parse_timestamp(value):
    """Parse nonnegative seconds, Russian units, mm:ss or hh:mm:ss."""
    if isinstance(value, bool):
        raise ValueError('boolean is not a timestamp')
    if isinstance(value, (int, float)):
        try:
            result = float(value)
        except OverflowError as error:
            raise ValueError('timestamp exceeds finite numeric range') from error
    elif isinstance(value, str):
        text = value.strip().lower()
        if re.fullmatch(r'\d+(?:\.\d+)?', text):
            result = float(text)
        elif re.fullmatch(r'\d+:\d{2}(?::\d{2}(?:\.\d+)?)?|\d+:\d{2}\.\d+', text):
            parts = [float(x) for x in text.split(':')]
            if any(x >= 60 for x in parts[1:]):
                raise ValueError('timestamp component out of range')
            result = 0.0
            for part in parts:
                result = result * 60 + part
        else:
            match = re.fullmatch(r'(?:(\d+)\s*(?:минута|минуты|минут|мин|м)\s*)?(?:(\d+(?:\.\d+)?)\s*(?:секунда|секунды|секунд|сек|с))?', text)
            if not match or not any(match.groups()):
                raise ValueError('malformed timestamp')
            minutes, seconds = match.groups()
            if minutes and seconds and float(seconds) >= 60:
                raise ValueError('seconds component out of range')
            result = float(minutes or 0) * 60 + float(seconds or 0)
    else:
        raise ValueError('timestamp must be a number or string')
    if not math.isfinite(result) or result < 0:
        raise ValueError('timestamp must be finite and nonnegative')
    return result
