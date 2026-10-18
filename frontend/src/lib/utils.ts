import { isJpegBase64 } from '../api-contract';
import { API_BASE_URL } from '../config';

export function formatTime(seconds: number): string {
  const minutes = Math.floor(seconds / 60)
    .toString()
    .padStart(2, '0');
  const remainder = Math.floor(seconds % 60)
    .toString()
    .padStart(2, '0');
  return `${minutes}:${remainder}`;
}

export function safeUrl(value: string): string | undefined {
  try {
    const url = new URL(value, API_BASE_URL || window.location.origin);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : undefined;
  } catch {
    return undefined;
  }
}

export function safeImageUrl(value: string): string | undefined {
  const prefix = 'data:image/jpeg;base64,';
  if (value.startsWith(prefix)) {
    return isJpegBase64(value.slice(prefix.length)) ? value : undefined;
  }
  return safeUrl(value);
}

export function delay(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    signal.throwIfAborted();

    function cleanup() {
      clearTimeout(timer);
      signal.removeEventListener('abort', handleAbort);
    }

    function handleAbort() {
      cleanup();
      reject(new DOMException('Отменено', 'AbortError'));
    }

    const timer = setTimeout(() => {
      cleanup();
      resolve();
    }, milliseconds);

    signal.addEventListener('abort', handleAbort, { once: true });
  });
}
