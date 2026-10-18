import type { AnalysisJob, AnalysisResult, Rule, Violation } from './types';

const VERDICTS = {
  violation: 'Обнаружены признаки возможных нарушений',
  no_violation: 'Нарушение не обнаружено',
} as const;

const STAGES: Record<string, string> = {
  queued: 'Видео в очереди',
  decode: 'Подготовка видеокадров',
  detect: 'Поиск объектов и действий',
  rules: 'Проверка признаков нарушений',
  verify: 'Проверка найденных эпизодов',
  done: 'Проверка завершена',
};

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isNonnegativeNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0;
}

/** Only JPEG data returned by the API may become a data URL; SVG/HTML are never accepted. */
export function isJpegBase64(value: unknown): value is string {
  return (
    typeof value === 'string' &&
    value.length % 4 === 0 &&
    /^\/9j\/[A-Za-z0-9+/]*={0,2}$/.test(value)
  );
}

function parseRule(value: unknown): Rule | undefined {
  if (value === undefined || value === null) return undefined;
  if (!isRecord(value) || typeof value.document !== 'string') {
    throw new Error('API вернул некорректное нормативное основание.');
  }

  const rule: Rule = { document: value.document };
  for (const field of ['clause', 'quote', 'url'] as const) {
    const text = value[field];
    if (text === undefined || text === null) continue;
    if (typeof text !== 'string') throw new Error('API вернул некорректное нормативное основание.');
    rule[field] = text;
  }
  return rule;
}

function parseViolation(value: unknown): Violation {
  if (
    !isRecord(value) ||
    typeof value.type !== 'string' ||
    !value.type.trim() ||
    typeof value.description !== 'string' ||
    !value.description.trim() ||
    !isNonnegativeNumber(value.timecode_sec) ||
    !Array.isArray(value.evidence_frames) ||
    value.evidence_frames.length < 1 ||
    value.evidence_frames.length > 3
  ) {
    throw new Error('API вернул некорректную карточку нарушения.');
  }

  const frames = value.evidence_frames.map((frame: unknown) => {
    if (
      !isRecord(frame) ||
      !isNonnegativeNumber(frame.timestamp_sec) ||
      !isJpegBase64(frame.image_base64)
    ) {
      throw new Error('API вернул некорректный JPEG-кадр.');
    }
    return `data:image/jpeg;base64,${frame.image_base64}`;
  });

  const violation: Violation = {
    type: value.type,
    timecode_sec: value.timecode_sec,
    description: value.description,
    evidence_frames: frames,
    rule: parseRule(value.rule),
  };

  for (const field of ['t_start', 't_end', 'confidence'] as const) {
    const number = value[field];
    if (number === undefined || number === null) continue;
    if (!isNonnegativeNumber(number) || (field === 'confidence' && number > 1)) {
      throw new Error('API вернул некорректные параметры эпизода.');
    }
    violation[field] = number;
  }
  return violation;
}

/** Convert the team's К1 wire contract into the UI's display model. */
export function validateResult(value: unknown): AnalysisResult {
  if (
    !isRecord(value) ||
    (value.verdict !== 'violation' && value.verdict !== 'no_violation') ||
    !Array.isArray(value.violations)
  ) {
    throw new Error('API вернул результат в неизвестном формате.');
  }
  if ((value.verdict === 'no_violation') !== (value.violations.length === 0)) {
    throw new Error('Вердикт API не согласован со списком нарушений.');
  }
  return {
    verdict: VERDICTS[value.verdict],
    violations: value.violations.map(parseViolation),
  };
}

export function parseJob(value: unknown): AnalysisJob {
  if (!isRecord(value) || typeof value.job_id !== 'string' || !value.job_id) {
    throw new Error('API не вернул идентификатор задания.');
  }
  if (value.status === 'done') {
    return { status: 'done', result: validateResult(value) };
  }
  if (value.status === 'failed') {
    if (value.error != null && typeof value.error !== 'string') {
      throw new Error('API вернул некорректный текст ошибки.');
    }
    return { status: 'failed', error: value.error || 'Не удалось обработать видео.' };
  }
  if (value.status !== 'processing') throw new Error('Неизвестный статус задания.');
  if (value.stage != null && typeof value.stage !== 'string') {
    throw new Error('API вернул некорректный этап обработки.');
  }
  if (value.progress != null && (!isNonnegativeNumber(value.progress) || value.progress > 1)) {
    throw new Error('API вернул некорректный прогресс обработки.');
  }
  return {
    status: 'processing',
    stage:
      typeof value.stage === 'string'
        ? STAGES[value.stage] || 'Анализ видео выполняется'
        : undefined,
    progress: typeof value.progress === 'number' ? value.progress : undefined,
  };
}
