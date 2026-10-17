"""К3 reader: violation types from kb/violations.yaml, validated once and cached.

A type keeps its `rule` only when Олжас has verified it (status=verified and verified_by set).
Anything else is still a known type, but without a normative reference — the pipeline then says
the basis is being clarified instead of citing a document nobody checked.
"""

import functools
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

log = logging.getLogger(__name__)

KB_PATH = Path(__file__).resolve().parent.parent / "kb" / "violations.yaml"
VERIFIED = "verified"
STATUSES = frozenset({"draft", "ready", VERIFIED})
REQUIRED_FIELDS = ("id", "name", "summary", "status")
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class KBError(ValueError):
    """kb/violations.yaml does not match the К3 format."""


@dataclass(frozen=True)
class RuleText:
    document: str
    number_date: str | None
    clause: str | None
    quote: str | None
    url: str | None

    @property
    def basis(self) -> str:
        """Short reference for the closing «Основание: …» sentence."""
        head = self.number_date or self.document
        return f"{head}, {self.clause}" if self.clause else head

    def as_ref(self) -> dict[str, str | None]:
        """К1 `rule` object (RuleRef)."""
        return {"document": self.document, "clause": self.clause, "quote": self.quote, "url": self.url}


@dataclass(frozen=True)
class ViolationType:
    id: str
    name: str
    summary: str
    status: str
    rule: RuleText | None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _parse_rule(type_id: str, status: str, raw: Any) -> RuleText | None:
    if status != VERIFIED:
        return None
    raw = raw if isinstance(raw, Mapping) else {}
    document, verified_by = _text(raw.get("document")), _text(raw.get("verified_by"))
    if not document or not verified_by:
        log.warning("kb: %s is marked verified but its rule lacks document/verified_by — emitted without rule", type_id)
        return None
    return RuleText(
        document=document,
        number_date=_text(raw.get("number_date")),
        clause=_text(raw.get("clause")),
        quote=_text(raw.get("quote")),
        url=_text(raw.get("url")),
    )


def _parse_entry(index: int, raw: Any) -> ViolationType:
    if not isinstance(raw, Mapping):
        raise KBError(f"violations[{index}] must be a mapping")
    missing = [f for f in REQUIRED_FIELDS if not _text(raw.get(f))]
    if missing:
        raise KBError(f"violations[{index}] misses required field(s): {', '.join(missing)}")
    type_id, status = _text(raw["id"]), _text(raw["status"])
    if not _ID_RE.match(type_id):
        raise KBError(f"violations[{index}]: id {type_id!r} is not snake_case")
    if status not in STATUSES:
        raise KBError(f"{type_id}: unknown status {status!r}, expected one of {sorted(STATUSES)}")
    return ViolationType(
        id=type_id,
        name=_text(raw["name"]),
        summary=_text(raw["summary"]),
        status=status,
        rule=_parse_rule(type_id, status, raw.get("rule")),
    )


def parse_catalog(raw: Any) -> Mapping[str, ViolationType]:
    entries = raw.get("violations") if isinstance(raw, Mapping) else None
    if not isinstance(entries, list):
        raise KBError("top-level `violations` list is missing")
    types: dict[str, ViolationType] = {}
    for index, item in enumerate(entries):
        vt = _parse_entry(index, item)
        if vt.id in types:
            raise KBError(f"duplicate violation id {vt.id!r}")
        types[vt.id] = vt
    return MappingProxyType(types)


def load_catalog(path: Path | str = KB_PATH) -> Mapping[str, ViolationType]:
    with open(path, encoding="utf-8") as fh:
        return parse_catalog(yaml.safe_load(fh))


@functools.cache
def catalog() -> Mapping[str, ViolationType]:
    """The repository catalog, parsed on first use."""
    types = load_catalog(KB_PATH)
    unverified = sorted(t.id for t in types.values() if t.rule is None)
    log.info("kb: %d violation types loaded, %d without verified rule: %s", len(types), len(unverified), unverified)
    return types
