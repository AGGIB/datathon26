"""Local intake utilities; errors never include private values."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlsplit


def require(condition, code):
    if not condition:
        raise ValueError(code)


def text(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except OverflowError:
        return False


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError('CLI_ARGUMENTS_INVALID')


def https(value):
    require(text(value), 'HTTPS_URL_REQUIRED')
    parsed = urlsplit(value)
    require(parsed.scheme == 'https' and parsed.hostname and not parsed.username
            and not parsed.password, 'PUBLIC_HTTPS_URL_REQUIRED')
    return value


def local_path(root, value):
    base = Path(root).absolute()
    require(base.resolve() == base, 'PRIVATE_ROOT_SYMLINK')
    path = Path(value)
    path = (path if path.is_absolute() else base / path).resolve()
    require(path.is_relative_to(base) and path != base, 'PATH_OUTSIDE_PRIVATE_ROOT')
    return path


def active_path(root, value):
    path = local_path(root, value)
    parts = {part.casefold() for part in path.relative_to(Path(root).absolute()).parts}
    require(not {'excluded', 'quarantine'} & parts, 'EXCLUDED_INPUT_OR_OUTPUT')
    return path


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'DUPLICATE_JSON_KEY')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def value_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def binary(value, name):
    path = shutil.which(value or name)
    require(path is not None, name.upper() + '_NOT_FOUND')
    return path
