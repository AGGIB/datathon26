import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createAnalysisRunGuard, validateVideoFile } from '../src/lib/analysis-state.ts';

const maxBytes = 200 * 1024 * 1024;
const deferred = () => {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
};

test('validates MOV/MP4, the exact size boundary and explicit clearing', () => {
  for (const name of ['synthetic.mov', 'synthetic.MP4']) {
    assert.equal(validateVideoFile({ name, size: maxBytes }, maxBytes), '');
  }
  assert.equal(validateVideoFile(undefined, maxBytes), '');
});

test('rejects invalid replacements and empty or oversized files without mutating them', () => {
  for (const file of [
    { name: 'synthetic.mp4.exe', size: 1 },
    { name: 'synthetic.webm', size: 1 },
    { name: 'synthetic.mp4', size: 0 },
    { name: 'synthetic.mp4', size: maxBytes + 1 },
    { name: 'synthetic.mp4', size: NaN },
    { name: 'synthetic.mp4', size: Infinity },
    { name: 'synthetic.mp4', size: -1 },
  ]) {
    const original = { ...file };
    assert.notEqual(validateVideoFile(file, maxBytes), '');
    assert.deepEqual(file, original);
  }
});

test('reset or cancellation prevents a pending success from restoring a result', async () => {
  const guard = createAnalysisRunGuard();
  const run = guard.begin();
  const response = deferred();
  let state = 'upload';
  const waiting = response.promise.then(() => run.commit(() => (state = 'done')));
  guard.invalidate();
  response.resolve();
  assert.equal(await waiting, false);
  assert.equal(state, 'upload');
  assert.equal(run.isCurrent(), false);
});

test('late progress and completion from an older run cannot overwrite the new run', async () => {
  const guard = createAnalysisRunGuard();
  const oldRun = guard.begin();
  const oldResponse = deferred();
  const oldWaiting = oldResponse.promise.then(() =>
    oldRun.commit(() => (state = { progress: 100, result: 'old' })),
  );
  let state = { progress: 0, result: undefined };
  const currentRun = guard.begin();
  assert.equal(
    currentRun.commit(() => (state = { progress: 25, result: undefined })),
    true,
  );
  assert.equal(
    oldRun.commit(() => (state.progress = 99)),
    false,
  );
  oldResponse.resolve();
  assert.equal(await oldWaiting, false);
  assert.deepEqual(state, { progress: 25, result: undefined });
  assert.equal(
    currentRun.commit(() => (state.result = 'new')),
    true,
  );
  assert.equal(state.result, 'new');
});

test('a late failure cannot replace a successful newer result with an error', async () => {
  const guard = createAnalysisRunGuard();
  const oldRun = guard.begin();
  const response = deferred();
  let error = '';
  const waiting = response.promise.catch(() => oldRun.commit(() => (error = 'old error')));
  const currentRun = guard.begin();
  response.reject(new Error('synthetic failure'));
  assert.equal(await waiting, false);
  assert.equal(error, '');
  assert.equal(currentRun.isCurrent(), true);
});

test('invalidation happens before synchronous abort callbacks may write', () => {
  const guard = createAnalysisRunGuard();
  const run = guard.begin();
  const controller = new AbortController();
  let progress = 0;
  controller.signal.addEventListener('abort', () => run.commit(() => (progress = 100)));
  guard.invalidate();
  controller.abort();
  assert.equal(progress, 0);
});

test('an obsolete finalizer does not invalidate the active run', () => {
  const guard = createAnalysisRunGuard();
  const oldRun = guard.begin();
  const currentRun = guard.begin();
  if (oldRun.isCurrent()) guard.invalidate();
  assert.equal(currentRun.isCurrent(), true);
  guard.invalidate();
  assert.equal(
    currentRun.commit(() => assert.fail('completed run wrote again')),
    false,
  );
});
