import assert from 'node:assert/strict';
import { test } from 'node:test';

let sequence = 0;
const freshConfig = () => import(`../src/config.ts?storage-test=${sequence++}`);
const blocked = () => {
  throw new DOMException('Synthetic restricted storage', 'SecurityError');
};

async function withWindow(windowValue, run) {
  const previous = Object.getOwnPropertyDescriptor(globalThis, 'window');
  Object.defineProperty(globalThis, 'window', { configurable: true, value: windowValue });
  try {
    await run(await freshConfig());
  } finally {
    if (previous) Object.defineProperty(globalThis, 'window', previous);
    else delete globalThis.window;
  }
}

function storageFixture() {
  const values = new Map();
  return {
    values,
    storage: {
      getItem: (key) => values.get(key) ?? null,
      setItem: (key, value) => values.set(key, value),
      removeItem: (key) => values.delete(key),
    },
  };
}

test('stored trimmed key retains precedence and clearing restores the default', async () => {
  const { storage, values } = storageFixture();
  values.set('forte_api_key', '  stored_synthetic_key  ');
  await withWindow({ localStorage: storage }, async (config) => {
    assert.equal(config.getApiKey(), 'stored_synthetic_key');
    config.setApiKey('  changed_synthetic_key  ');
    assert.equal(values.get('forte_api_key'), 'changed_synthetic_key');
    assert.equal(config.getApiKey(), 'changed_synthetic_key');
    config.setApiKey('  ');
    assert.equal(values.has('forte_api_key'), false);
    assert.equal(config.getApiKey(), config.DEFAULT_API_KEY);
  });
});

test('a localStorage getter SecurityError does not break reads or entered keys', async () => {
  const windowValue = {};
  Object.defineProperty(windowValue, 'localStorage', { get: blocked });
  await withWindow(windowValue, async (config) => {
    assert.equal(config.getApiKey(), config.DEFAULT_API_KEY);
    assert.doesNotThrow(() => config.setApiKey('  page_synthetic_key  '));
    assert.equal(config.getApiKey(), 'page_synthetic_key');
    config.setApiKey('');
    assert.equal(config.getApiKey(), config.DEFAULT_API_KEY);
  });
});

test('blocked getItem falls back without a UI crash', async () => {
  await withWindow({ localStorage: { getItem: blocked } }, async (config) => {
    assert.equal(config.getApiKey(), config.DEFAULT_API_KEY);
  });
});

test('failed writes keep the entered effective key and failed clear ignores stale storage', async () => {
  const storage = {
    getItem: () => 'stale_synthetic_key',
    setItem: blocked,
    removeItem: blocked,
  };
  await withWindow({ localStorage: storage }, async (config) => {
    assert.equal(config.getApiKey(), 'stale_synthetic_key');
    assert.doesNotThrow(() => config.setApiKey('page_synthetic_key'));
    assert.equal(config.getApiKey(), 'page_synthetic_key');
    assert.doesNotThrow(() => config.setApiKey(''));
    assert.equal(config.getApiKey(), config.DEFAULT_API_KEY);
  });
});

test('a successful later write restores normal storage precedence', async () => {
  const { storage, values } = storageFixture();
  const write = storage.setItem;
  storage.setItem = blocked;
  await withWindow({ localStorage: storage }, async (config) => {
    config.setApiKey('memory_synthetic_key');
    assert.equal(config.getApiKey(), 'memory_synthetic_key');
    storage.setItem = write;
    config.setApiKey('persisted_synthetic_key');
    assert.equal(config.getApiKey(), 'persisted_synthetic_key');
    values.set('forte_api_key', 'external_synthetic_key');
    assert.equal(config.getApiKey(), 'external_synthetic_key');
  });
});

test('missing storage still permits a page-local override and an explicit clear', async () => {
  await withWindow({}, async (config) => {
    config.setApiKey('page_synthetic_key');
    assert.equal(config.getApiKey(), 'page_synthetic_key');
    config.setApiKey('');
    assert.equal(config.getApiKey(), config.DEFAULT_API_KEY);
  });
});
