"""К3 reader: violation types from kb/violations.yaml, validated once and cached.

The kb owns the wording; code only fills placeholders. A `description_template` in the contract format
uses {observation} plus any of PLACEHOLDERS and ends with one of VERIFIED_ENDINGS (type with a verified
norm: the document title or, preferred by the lead, its number and date) or DRAFT_ENDING (no verified norm,
case-insensitive).

Failure policy: structural errors (bad yaml shape, unknown placeholder in a contract template,
{rule_number_date} used while rule.number_date is empty) raise KBError at load, and the backend loads the catalog at startup, so a broken kb stops the boot. Problems that
only concern one type's wording (ending does not match the current state of the norm, a typo such as
{observaton}) are logged as ERROR and that type gets FALLBACK_TEMPLATES; jobs keep running. Templates not
yet in the contract format (the generic «<что видно>» one) get the fallback with a warning. A type keeps its
`rule` only when the norm is verified; otherwise no reference is cited at all.
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
VERIFIED_ENDING_NUMBER_DATE = "Основание: {rule_number_date}, {rule_clause}."  # act type, number, date
VERIFIED_ENDINGS = (VERIFIED_ENDING, VERIFIED_ENDING_NUMBER_DATE)
DRAFT_ENDING = "Нормативное основание уточняется."
# Generic wording keeps the caveat that kb templates carry: the requirement may not apply / may not exist.
FALLBACK_TEMPLATES = MappingProxyType({
    VERIFIED: "На {t} с {observation}. Это может указывать на несоблюдение требования, указанного в основании, "
    "если оно применимо. " + VERIFIED_ENDING,
    "draft": "На {t} с {observation}. Это может указывать на несоблюдение применимого порядка, если он установлен. "
    + DRAFT_ENDING,
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


def _fields(template: str) -> list[tuple[str, str, str | None]]:
    """(name, format_spec, conversion) of every placeholder; ValueError on broken braces."""
    return [(name, spec or "", conv) for _lit, name, spec, conv in string.Formatter().parse(template) if name is not None]


def _structural_names(type_id: str, template: str) -> set[str]:
    """Placeholders of a contract template. Anything the renderer cannot fill is a kb error (fail fast)."""
    try:
        fields = _fields(template)
    except ValueError as exc:
        raise KBError(f"{type_id}: description_template is not a valid template ({exc})") from None
    for name, spec, conversion in fields:
        if spec or conversion or name not in PLACEHOLDERS:
            raise KBError(f"{type_id}: unknown placeholder {{{name}}} in description_template, "
                          f"allowed: {sorted(PLACEHOLDERS)}")
    return {name for name, _, _ in fields}


def _state_problem(template: str, names: set[str], rule: RuleText | None) -> str | None:
    """Mismatch between the wording and the current state of the norm (owned by different people)."""
    if rule is None:
        if not template.lower().endswith(DRAFT_ENDING.lower()):
            return f"the type has no verified norm, so its template must end with «{DRAFT_ENDING}»"
        if names & RULE_PLACEHOLDERS:
            return "the template cites rule fields, but the type has no verified norm"
        return None
    if not template.endswith(VERIFIED_ENDINGS):
        return "the type has a verified norm, so its template must end with " + " or ".join(
            f"«{ending}»" for ending in VERIFIED_ENDINGS)
    return None


def _without_missing_clause(template: str, rule: RuleText) -> str:
    return template if rule.clause else template.replace(", {rule_clause}", "")


def _fallback(rule: RuleText | None) -> str:
    return _without_missing_clause(FALLBACK_TEMPLATES[VERIFIED], rule) if rule else FALLBACK_TEMPLATES["draft"]


def _unknown_names(template: str) -> list[str]:
    try:
        return sorted({name for name, _, _ in _fields(template)} - PLACEHOLDERS)
    except ValueError:
        return []


def _parse_template(type_id: str, raw: Any, rule: RuleText | None) -> tuple[str, bool]:
    """(template, is_legacy). Errors that only affect this type's wording are logged and fall back."""
    template = _text(raw)
    if template is None or "{observation}" not in template:
        unknown = _unknown_names(template or "")
        if unknown:  # most likely a typo such as {observaton}
            log.error("kb: %s: description_template has unknown placeholder(s) %s and no {observation} — "
                      "generic wording used", type_id, unknown)
            return _fallback(rule), False
        return _fallback(rule), True
    names = _structural_names(type_id, template)
    if rule is not None and "rule_number_date" in names and not rule.number_date:
        raise KBError(f"{type_id}: description_template uses {{rule_number_date}}, but rule.number_date is empty")
    problem = _state_problem(template, names, rule)
    if problem:
        log.error("kb: %s: %s — generic wording used for this type", type_id, problem)
        return _fallback(rule), False
    return (_without_missing_clause(template, rule) if rule else template), False


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
    template, is_legacy = _parse_template(type_id, raw.get("description_template"), rule)
    vtype = ViolationType(
        id=type_id,
        name=_text(raw["name"]),
        summary=_text(raw.get("summary")),
        status=status,
        rule=rule,
        template=template,
        enabled=raw.get("enabled", True) is not False,
    )
    return vtype, is_legacy


def parse_catalog(raw: Any) -> Mapping[str, ViolationType]:
    entries = raw.get("violations") if isinstance(raw, Mapping) else None
    if not isinstance(entries, list):
        raise KBError("top-level `violations` list is missing")
    types: dict[str, ViolationType] = {}
    legacy: list[str] = []
    for index, item in enumerate(entries):
        vtype, is_legacy = _parse_entry(index, item)
        if vtype.id in types:
            raise KBError(f"duplicate violation id {vtype.id!r}")
        types[vtype.id] = vtype
        if is_legacy:
            legacy.append(vtype.id)
    if legacy:
        log.warning("kb: %d type(s) have no description_template in the contract format, generic wording used: %s",
                    len(legacy), legacy)
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
