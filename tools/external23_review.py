"""Generate a private offline review page; never opens or copies media."""

import argparse
import html
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote
from external23_policy import source_usable


ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "external23"


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError("review_arguments_invalid")


def require(condition, code):
    if not condition:
        raise ValueError(code)


def private_path(value, base=ROOT):
    path = Path(value)
    path = (path if path.is_absolute() else base / path).resolve()
    require(path.is_relative_to(PRIVATE), "review_path_outside_private")
    require(not {'excluded', 'quarantine'} & set(path.relative_to(PRIVATE).parts), "review_excluded_path")
    return path


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def escaped(value):
    return html.escape(str(value), quote=True)


def script_json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False).replace(
        "<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def render(index, annotations, model_document, output):
    require(isinstance(index, dict) and isinstance(index.get("records"), list),
            "review_index_invalid")
    require(isinstance(annotations, dict) and annotations.get("schema_version") == 1
            and isinstance(annotations.get("clips"), list), "review_annotations_invalid")
    records, drafts = {}, {}
    for record in index["records"]:
        require(isinstance(record, dict) and isinstance(record.get("clip_id"), str)
                and bool(record["clip_id"]) and record["clip_id"] not in records,
                "review_record_invalid")
        source_usable(record)
        require(record['clip_id'].startswith('ext23-'), 'review_legacy_id_excluded')
        records[record["clip_id"]] = record
    for draft in annotations["clips"]:
        require(isinstance(draft, dict) and isinstance(draft.get("clip_id"), str)
                and draft["clip_id"] not in drafts, "review_annotation_clip_invalid")
        drafts[draft["clip_id"]] = draft
    require(set(records) == set(drafts), "review_clip_set_mismatch")
    model_items = model_document
    if isinstance(model_document, dict):
        model_items = model_document.get("items", model_document.get(
            "results", model_document.get("raw_results")))
    require(isinstance(model_items, list), "review_model_results_invalid")
    suggestions = {clip_id: [] for clip_id in records}
    for item in model_items:
        require(isinstance(item, dict) and isinstance(item.get("clip_id"), str)
                and item["clip_id"] in records, "review_model_item_invalid")
        suggestions[item["clip_id"]].append(item)
    cards = []
    for position, (clip_id, record) in enumerate(records.items()):
        require(isinstance(record.get("clip_path"), str) and bool(record["clip_path"]),
                "review_clip_path_invalid")
        media = private_path(record["clip_path"], PRIVATE)
        source = quote(Path(os.path.relpath(media, output.parent)).as_posix(), safe="/.")
        provenance = {key: record.get(key) for key in (
            "clip_id", "ffprobe", "source_group", "camera_id", "source_url", "license",
            "attribution", "source_offsets", "source_context", "source_sha256", "clip_sha256")}
        cards.append(f'''<section class="card" data-position="{position}">
<h2>{escaped(clip_id)}</h2>
<video controls preload="metadata" src="{escaped(source)}"></video>
<p><label>Переход, секунды клипа: <input class="seek" type="number" min="0" step="0.1"></label>
<button class="seek-button" type="button">Перейти</button> <span class="seek-status" role="status"></span></p>
<details><summary>Происхождение и технические данные</summary><pre>{escaped(json_text(provenance))}</pre></details>
<details><summary>Непроверенные предложения модели ({len(suggestions[clip_id])})</summary>
<p>local_model / pending_review. Это подсказки для локальной проверки, не разметка человека.</p>
<pre>{escaped(json_text(suggestions[clip_id]))}</pre></details>
<label class="editor-label" for="draft-{position}">Редактировать JSON разметки клипа:</label>
<textarea id="draft-{position}" class="draft" spellcheck="false">{escaped(json_text(drafts[clip_id]))}</textarea>
</section>''')
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; media-src 'self' file:; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Локальная проверка external23</title>
<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem;background:#f5f6f8;color:#18202c}h1{font-size:1.7rem}.notice,.card{padding:1.1rem;background:white;border:1px solid #ccd3dd;border-radius:8px;margin:1rem 0}video{display:block;width:100%;max-height:560px;background:#111}pre{white-space:pre-wrap;overflow-wrap:anywhere}textarea{display:block;box-sizing:border-box;width:100%;min-height:22rem;font:14px ui-monospace,monospace;margin:.5rem 0}button,input{font:inherit;padding:.35rem}.editor-label{display:block;margin-top:1rem}summary{cursor:pointer}#download-status{white-space:pre-wrap}</style>
<h1>Локальная проверка external23</h1>
<div class="notice"><p>Черновик: pending_review. Страница работает локально и не отправляет данные в сеть.
Видео не запускаются автоматически. Время событий указывается в секундах клипа.</p>
<p><strong>Происхождение:</strong> source_human — опубликованная исходная разметка;
local_model — предложение локальной модели; reviewed_human — фактическая проверка человеком с его именем и датой.
Ни просмотр страницы, ни скачивание JSON не подтверждают проверку человеком.</p>
<p>Действия и proxy-сцены не доказывают банковское нарушение. legal_gold по умолчанию null.
Изменения JSON требуют внешней проверки инструментом check-annotations; страница её не заменяет.</p>
<p>JSON содержит только показанные клипы. Сохраните поднабор отдельно и объедините по clip_id
с общим annotations.json; затем проверьте полный файл командой check-annotations.</p>
<button id="download" type="button">Скачать review-subset.annotations.json</button>
<p id="download-status" role="status"></p></div>
__CARDS__
<script type="application/json" id="annotations-data">__ANNOTATIONS__</script>
<script>
"use strict";
const original = JSON.parse(document.getElementById("annotations-data").textContent);
const cards = Array.from(document.querySelectorAll(".card"));
for (const card of cards) {
  card.querySelector(".seek-button").addEventListener("click", () => {
    const video = card.querySelector("video");
    const input = card.querySelector(".seek");
    const time = Number(input.value);
    const valid = input.value.trim() !== "" && Number.isFinite(time) && time >= 0 && Number.isFinite(video.duration) && time <= video.duration;
    card.querySelector(".seek-status").textContent = valid ? "Позиция установлена" : "Укажите время в пределах загруженного клипа";
    if (valid) { video.pause(); video.currentTime = time; }
  });
}
document.getElementById("download").addEventListener("click", () => {
  const status = document.getElementById("download-status");
  try {
    const edited = JSON.parse(JSON.stringify(original));
    edited.clips = cards.map(card => {
      const position = Number(card.dataset.position);
      const draft = JSON.parse(card.querySelector(".draft").value);
      const expected = original.clips.find(clip => clip.clip_id === cards[position].querySelector("h2").textContent);
      if (!draft || Array.isArray(draft) || typeof draft !== "object" || !expected || draft.clip_id !== expected.clip_id) throw new Error("invalid");
      return draft;
    });
    edited.status = "pending_review";
    const url = URL.createObjectURL(new Blob([JSON.stringify(edited, null, 2) + "\\n"], {type: "application/json"}));
    const link = document.createElement("a"); link.href = url; link.download = "review-subset.annotations.json";
    document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    status.textContent = "Поднабор скачан как pending_review. Сохраните отдельно, объедините по clip_id с общим annotations.json и проверьте полный файл.";
  } catch (_) { status.textContent = "Проверьте JSON и неизменность clip_id. Файл не создан."; }
});
</script></html>'''
    page = page.replace("__ANNOTATIONS__", script_json(annotations), 1).replace(
        "__CARDS__", "\n".join(cards), 1)
    return page, len(records), len(model_items)


def main():
    parser = SafeParser(description=__doc__)
    parser.add_argument("--index", default="data/external23/intake.json")
    parser.add_argument("--annotations", default="data/external23/annotations.json")
    parser.add_argument("--model-results")
    parser.add_argument("--output", default="data/external23/review/index.html")
    try:
        args = parser.parse_args()
        index_path, annotation_path = private_path(args.index), private_path(args.annotations)
        output = private_path(args.output)
        require(output.suffix.lower() == ".html", "review_output_invalid")
        model_path = private_path(args.model_results) if args.model_results else None
        require(output not in (index_path, annotation_path, model_path), "review_output_overlaps_input")
        model = read_json(model_path) if model_path else []
        page, clips, items = render(read_json(index_path), read_json(annotation_path), model, output)
        output.parent.mkdir(parents=True, exist_ok=True)
        private_path(output)
        import tempfile
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(page)
        temporary.replace(output)
    except (OSError, ValueError, TypeError, KeyError):
        print("ERROR: review_input_or_output_invalid", file=sys.stderr)
        return 1
    print(f"OK: review clips={clips} model_items={items}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
