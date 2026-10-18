const env: Record<string, string | undefined> =
  (import.meta as unknown as { env?: Record<string, string | undefined> }).env ?? {};
export const API_BASE_URL = (env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
// Production uses the same origin by default. Demo is explicit for published builds.
export const IS_DEMO =
  env.VITE_DEMO_MODE === 'true' ||
  (env.VITE_DEMO_MODE !== 'false' && Boolean(env.DEV) && API_BASE_URL.length === 0);
export const MAX_FILE_SIZE = 200 * 1024 * 1024;
export const POLL_INTERVAL_MS = 2_000;
export const ANALYSIS_TIMEOUT_MS = 5 * 60 * 1_000;
export const DEFAULT_API_KEY = 'ui_forte26_4a6e8b0d2c';
// Used only when this page cannot persist an explicitly entered override.
// Empty string records an explicit clear even if removeItem was blocked.
let memoryApiKey: string | undefined;

function configuredApiKey(): string {
  return env.VITE_API_KEY?.trim() || DEFAULT_API_KEY;
}

export function getApiKey(): string {
  if (memoryApiKey !== undefined) return memoryApiKey || configuredApiKey();
  try {
    const storage = typeof window !== 'undefined' ? window.localStorage : undefined;
    const stored = storage?.getItem('forte_api_key')?.trim();
    if (stored) return stored;
  } catch {
    // Restricted browsing can throw even when accessing window.localStorage.
  }
  return configuredApiKey();
}

export function setApiKey(key: string): void {
  const normalized = key.trim();
  memoryApiKey = normalized;
  try {
    const storage = typeof window !== 'undefined' ? window.localStorage : undefined;
    if (!storage) return;
    if (normalized) {
      storage.setItem('forte_api_key', normalized);
    } else {
      storage.removeItem('forte_api_key');
    }
    memoryApiKey = undefined;
  } catch {
    // The effective key remains the user's override for this page lifetime.
  }
}
