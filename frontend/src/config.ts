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
export const DEFAULT_API_KEY = '';
let sessionApiKey: string | undefined;

export function getApiKey(): string {
  if (sessionApiKey !== undefined) return sessionApiKey;
  try {
    if (typeof window !== 'undefined') {
      const stored = window.localStorage.getItem('forte_api_key')?.trim();
      if (stored) return stored;
    }
  } catch {
    // Storage can be unavailable; the key can still be entered for this session.
  }
  const viteKey = (env.VITE_API_KEY as string | undefined)?.trim();
  if (viteKey) return viteKey;
  return DEFAULT_API_KEY;
}

export function setApiKey(key: string): void {
  sessionApiKey = key.trim() || undefined;
  try {
    if (typeof window !== 'undefined') {
      if (sessionApiKey) {
        window.localStorage.setItem('forte_api_key', sessionApiKey);
      } else {
        window.localStorage.removeItem('forte_api_key');
      }
    }
  } catch {
    // Keep the in-memory value even when browser persistence is blocked.
  }
}
