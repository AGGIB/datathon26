"""К3 reader: violation types from kb/violations.yaml, validated once and cached.

The kb owns the wording; code only fills placeholders. A `description_template` in the contract format
uses {observation} plus any of PLACEHOLDERS and ends with VERIFIED_ENDING (type with a verified norm) or
DRAFT_ENDING (no verified norm). Such templates are validated strictly, so a typo fails at load time.
Templates not yet in that format (the generic «<что видно>» one, or prose without {observation}) are
replaced by FALLBACK_TEMPLATES with a warning. A type keeps its `rule` only when the norm is verified;
otherwise no reference is cited at all.
"""

import functools
import logging
import re
import string
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
REQUIRED_FIELDS = ("id", "name", "status")
CITATION_VERIFIED = "citation_verified"
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")

PLACEHOLDERS = frozenset({"t", "observation", "rule_document", "rule_clause", "rule_number_date"})
RULE_PLACEHOLDERS = frozenset({"rule_document", "rule_clause", "rule_number_date"})
VERIFIED_ENDING = "Основание: {rule_document}, {rule_clause}."
DRAFT_ENDING = "Нормативное основание уточняется."
FALLBACK_TEMPLATES = MappingProxyType({
    "verified": "На {t} с {observation}. Это может указывать на несоблюдение требования, указанного в основании. "
    + VERIFIED_ENDING,
    "draft": "На {t} с {observation}. Это может указывать на несоблюдение установленного порядка. " + DRAFT_ENDING,
})


class KBError(ValueError):
    """kb/violations.yaml does not match the К3 format."""


@dataclass(frozen=True)
class RuleText:
    document: str
    number_date: str | None
    clause: str | None
    quote: str | None
    url: str | None

    def as_ref(self) -> dict[str, str | None]:
        """К1 `rule` object (RuleRef)."""
        return {"document": self.document, "clause": self.clause, "quote": self.quote, "url": self.url}


@dataclass(frozen=True)
class ViolationType:
    id: str
    name: str
    summary: str | None
    status: str
    rule: RuleText | None
    template: str  # always in the contract format (validated or fallback)
    enabled: bool = True


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _citation_ok(raw: Mapping, rule: Mapping) -> bool:
    statuses = [s for s in (_text(raw.get("citation_status")), _text(rule.get("citation_status"))) if s]
    return all(s == CITATION_VERIFIED for s in statuses)


def _parse_rule(type_id: str, raw: Mapping) -> RuleText | None:
    if _text(raw.get("status")) != VERIFIED:
        return None
    rule = raw.get("rule") if isinstance(raw.get("rule"), Mapping) else {}
    document, verified_by = _text(rule.get("document")), _text(rule.get("verified_by"))
    if not document or not verified_by or not _citation_ok(raw, rule):
        log.warning("kb: %s is marked verified but its rule is not verified — emitted without rule", type_id)
        return None
    return RuleText(
        document=document,
        number_date=_text(rule.get("number_date")),
        clause=_text(rule.get("clause")),
        quote=_text(rule.get("quote")),
        url=_text(rule.get("url")),
    )


def _placeholders(type_id: str, template: str) -> set[str]:
    try:
        parsed = list(string.Formatter().parse(template))
    except ValueError as exc:
        raise KBError(f"{type_id}: description_template is not a valid template ({exc})") from None
    names = set()
    for _literal, name, spec, conversion in parsed:
        if name is None:
            continue
        if spec or conversion or name not in PLACEHOLDERS:
            raise KBError(f"{type_id}: unknown placeholder {{{name}}} in description_template, "
                          f"allowed: {sorted(PLACEHOLDERS)}")
        names.add(name)
    return names


def _check_contract(type_id: str, template: str, rule: RuleText | None) -> str:
    names = _placeholders(type_id, template)
    if rule is None:
        if not template.endswith(DRAFT_ENDING):
            raise KBError(f"{type_id}: template of a type without verified norm must end with «{DRAFT_ENDING}»")
        if names & RULE_PLACEHOLDERS:
            raise KBError(f"{type_id}: template references rule fields, but the type has no verified rule")
        return template
    if not template.endswith(VERIFIED_ENDING):
        raise KBError(f"{type_id}: template of a verified type must end with «{VERIFIED_ENDING}»")
    if "rule_number_date" in names and not rule.number_date:
        raise KBError(f"{type_id}: template uses {{rule_number_date}}, but rule.number_date is empty")
    return _without_missing_clause(template, rule)


def _without_missing_clause(template: str, rule: RuleText) -> str:
    return template if rule.clause else template.replace(", {rule_clause}", "")


def _parse_template(type_id: str, raw: Any, rule: RuleText | None) -> tuple[str, bool]:
    """(template, is_fallback)."""
    template = _text(raw)
    if template is not None and "{observation}" in template:
        return _check_contract(type_id, template, rule), False
    fallback = FALLBACK_TEMPLATES[VERIFIED if rule else "draft"]
    return (_without_missing_clause(fallback, rule) if rule else fallback), True


def _parse_entry(index: int, raw: Any) -> tuple[ViolationType, bool]:
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
    rule = _parse_rule(type_id, raw)
    template, is_fallback = _parse_template(type_id, raw.get("description_template"), rule)
    vtype = ViolationType(
        id=type_id,
        name=_text(raw["name"]),
        summary=_text(raw.get("summary")),
        status=status,
        rule=rule,
        template=template,
        enabled=raw.get("enabled", True) is not False,
    )
    return vtype, is_fallback


def parse_catalog(raw: Any) -> Mapping[str, ViolationType]:
    entries = raw.get("violations") if isinstance(raw, Mapping) else None
    if not isinstance(entries, list):
        raise KBError("top-level `violations` list is missing")
    types: dict[str, ViolationType] = {}
    fallbacks: list[str] = []
    for index, item in enumerate(entries):
        vtype, is_fallback = _parse_entry(index, item)
        if vtype.id in types:
            raise KBError(f"duplicate violation id {vtype.id!r}")
        types[vtype.id] = vtype
        if is_fallback:
            fallbacks.append(vtype.id)
    if fallbacks:
        log.warning("kb: %d type(s) have no description_template in the contract format, generic wording used: %s",
                    len(fallbacks), fallbacks)
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
