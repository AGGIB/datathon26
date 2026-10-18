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
  assert.deepEqual(job.result.violations[0].evidence_frames[0], {
    timestamp_sec: 22.4,
    image_url: `data:image/jpeg;base64,${jpeg}`,
  });
  assert.deepEqual(job.result.violations[0].rule, { document: 'Документ' });
});

test('preserves each of three actual evidence timestamps independently of the episode peak', () => {
  const timestamps = [18.125, 22.4, 27.875];
  const evidence_frames = timestamps.map((timestamp_sec) => ({
    timestamp_sec,
    image_base64: jpeg,
  }));
  const result = validateResult(done({ violations: [{ ...violation, evidence_frames }] }));
  assert.equal(result.violations[0].timecode_sec, 22.4);
  assert.deepEqual(
    result.violations[0].evidence_frames,
    timestamps.map((timestamp_sec) => ({
      timestamp_sec,
      image_url: `data:image/jpeg;base64,${jpeg}`,
    })),
  );
  assert.equal(evidence_frames[0].image_base64, jpeg);
});

test('keeps two episodes with the same peak as separate stable response identities', () => {
  const payload = done({
    violations: [
      { ...violation, description: 'Первый вымышленный эпизод.' },
      {
        ...violation,
        description: 'Второй вымышленный эпизод с тем же пиком.',
        evidence_frames: [{ timestamp_sec: 25.75, image_base64: jpeg }],
      },
    ],
  });
  const result = validateResult(payload);
  assert.equal(result.violations.length, 2);
  assert.equal(result.violations[0].timecode_sec, result.violations[1].timecode_sec);
  assert.notEqual(result.violations[0].id, result.violations[1].id);
  assert.deepEqual(
    validateResult(payload).violations.map((episode) => episode.id),
    result.violations.map((episode) => episode.id),
  );
  assert.equal(result.violations[0].evidence_frames[0].timestamp_sec, 22.4);
  assert.equal(result.violations[1].evidence_frames[0].timestamp_sec, 25.75);
});

test('rejects invalid per-frame timestamps before they can become seek targets', () => {
  for (const timestamp_sec of [-1, NaN, Infinity, '22.4', null]) {
    assert.throws(
      () =>
        validateResult(
          done({
            violations: [
              { ...violation, evidence_frames: [{ timestamp_sec, image_base64: jpeg }] },
            ],
          }),
        ),
      /JPEG/,
    );
  }
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

test('prototype property names remain unknown processing stages', () => {
  for (const stage of ['__proto__', 'constructor', 'toString', 'hasOwnProperty']) {
    const job = parseJob({ job_id: 'id', status: 'processing', stage });
    assert.equal(job.stage, 'Анализ видео выполняется');
    assert.equal(typeof job.stage, 'string');
  }
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

test('getApiKey defaults to UI key and respects localStorage override', async () => {
  const { DEFAULT_API_KEY, getApiKey, setApiKey } = await import('../src/config.ts');
  assert.equal(getApiKey(), DEFAULT_API_KEY);

  const storage = new Map();
  globalThis.window = {
    localStorage: {
      getItem: (k) => storage.get(k) || null,
      setItem: (k, v) => storage.set(k, String(v)),
      removeItem: (k) => storage.delete(k),
    },
  };

  try {
    assert.equal(getApiKey(), DEFAULT_API_KEY);
    setApiKey('custom_test_key_123');
    assert.equal(getApiKey(), 'custom_test_key_123');
    setApiKey('');
    assert.equal(getApiKey(), DEFAULT_API_KEY);
  } finally {
    delete globalThis.window;
  }
});

test('a response from another job cannot replace the current check', () => {
  assert.throws(
    () => parseJob({ job_id: 'other-job', status: 'processing' }, 'current-job'),
    /другой проверки/,
  );
  assert.equal(
    parseJob({ job_id: 'current-job', status: 'processing' }, 'current-job').status,
    'processing',
  );
});
