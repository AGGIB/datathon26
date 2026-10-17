const env = import.meta.env;
export const API_BASE_URL = (env.VITE_API_BASE_URL ?? '').replace(/\/$/, '');
// Production uses the same origin by default. Demo is explicit for published builds.
export const IS_DEMO =
  env.VITE_DEMO_MODE === 'true' ||
  (env.VITE_DEMO_MODE !== 'false' && env.DEV && API_BASE_URL.length === 0);
export const MAX_FILE_SIZE = 200 * 1024 * 1024;
export const POLL_INTERVAL_MS = 2_000;
export const ANALYSIS_TIMEOUT_MS = 5 * 60 * 1_000;
