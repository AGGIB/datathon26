"""Stable active runtime IDs; pending normative citations do not disable types."""
from pathlib import Path
from external23_common import read_json, require


def active_type_ids(path=None):
    path = path or Path(__file__).resolve().parents[1] / 'kb/type-ids.json'
    rows = [r for r in read_json(path)['types'] if r['catalog_section'] == 'violations']
    result = {r['id'] for r in rows}
    require(len(rows) == len(result) == 27, 'ACTIVE_CATALOG_IDS_INVALID')
    return result
