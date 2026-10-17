import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseJob, validateResult, isJpegBase64 } from '../src/api-contract.ts';

// JPEG marker bytes encoded as base64; image decoding belongs to the browser.
const jpeg = '/9j/2Q==';
const violation = {
  type: 'Тестовый эпизод',
  timecode_sec: 22.4,
  description: 'Наблюдаемое действие.',
  evidence_frames: [{ timestamp_sec: 22.4, image_base64: jpeg }],
  rule: { document: 'Документ', clause: null, quote: null, url: null },
};

const done = (overrides = {}) => ({
  job_id: 'job-123',
  status: 'done',
  verdict: 'violation',
  violations: [violation],
  ...overrides,
});

test('К1 done maps verdict, JPEG frames and nullable normative fields for the UI', () => {
  const job = parseJob(done());
  assert.equal(job.status, 'done');
  assert.equal(job.result.verdict, 'Обнаружены признаки возможных нарушений');
  assert.equal(job.result.violations[0].evidence_frames[0], `data:image/jpeg;base64,${jpeg}`);
  assert.deepEqual(job.result.violations[0].rule, { document: 'Документ' });
});

test('no_violation renders the required Russian verdict with no cards', () => {
  const job = parseJob(done({ verdict: 'no_violation', violations: [] }));
  assert.equal(job.result.verdict, 'Нарушение не обнаружено');
  assert.deepEqual(job.result.violations, []);
});

test('processing translates stage and preserves fractional progress', () => {
  assert.deepEqual(
    parseJob({ job_id: 'id', status: 'processing', stage: 'rules', progress: 0.5 }),
    {
      status: 'processing',
      stage: 'Проверка признаков нарушений',
      progress: 0.5,
    },
  );
  assert.equal(
    parseJob({ job_id: 'id', status: 'processing', stage: 'future-stage' }).stage,
    'Анализ видео выполняется',
  );
});

test('failed preserves the backend error', () => {
  assert.deepEqual(parseJob({ job_id: 'id', status: 'failed', error: 'Сбой декодирования' }), {
    status: 'failed',
    error: 'Сбой декодирования',
  });
});

test('rejects verdicts inconsistent with evidence', () => {
  assert.throws(() => validateResult(done({ violations: [] })), /не согласован/);
  assert.throws(() => validateResult(done({ verdict: 'no_violation' })), /не согласован/);
});

test('rejects non-JPEG frames, invalid timestamps and invalid frame counts', () => {
  for (const image of [
    '<svg onload="alert(1)">',
    'data:image/svg+xml;base64,PHN2Zz4=',
    'javascript:alert(1)',
  ]) {
    assert.equal(isJpegBase64(image), false);
    assert.throws(
      () =>
        validateResult(
          done({
            violations: [
              { ...violation, evidence_frames: [{ timestamp_sec: 0, image_base64: image }] },
            ],
          }),
        ),
      /JPEG/,
    );
  }
  for (const frames of [[], Array(4).fill(violation.evidence_frames[0])]) {
    assert.throws(() =>
      validateResult(done({ violations: [{ ...violation, evidence_frames: frames }] })),
    );
  }
  assert.throws(() => validateResult(done({ violations: [{ ...violation, timecode_sec: -1 }] })));
});

test('rejects malformed statuses and progress without crashing the view', () => {
  for (const progress of [-1, 1.5, '50%', NaN]) {
    assert.throws(() => parseJob({ job_id: 'id', status: 'processing', progress }), /прогресс/);
  }
  assert.throws(() => parseJob({ job_id: 'id', status: 'processing', stage: {} }), /этап/);
  assert.throws(() => parseJob({ job_id: 'id', status: 'failed', error: {} }), /ошибки/);
});
