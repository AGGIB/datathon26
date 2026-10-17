import { isRecord, parseJob } from './api-contract';
import { API_BASE_URL } from './config';
import type { AnalysisJob } from './types';

function getHttpErrorMessage(status: number): string {
  switch (status) {
    case 400:
      return 'Видео пустое или повреждено. Выберите другой файл.';
    case 404:
      return 'Задание не найдено. Запустите проверку заново.';
    case 411:
      return 'Сервер не смог определить размер загрузки. Повторите отправку.';
    case 401:
      return 'Сервис отклонил запрос. Обратитесь к команде API.';
    case 413:
      return 'Сервер отклонил файл: превышен размер.';
    case 415:
    case 422:
      return 'Сервер не смог прочитать видео. Поддерживаются MOV и MP4.';
    default:
      return `Ошибка сервера (${status}). Попробуйте ещё раз.`;
  }
}

export function uploadVideo(
  file: File,
  onProgress: (percent: number) => void,
  signal: AbortSignal,
): Promise<string> {
  return new Promise((resolve, reject) => {
    signal.throwIfAborted();
    const request = new XMLHttpRequest();
    request.open('POST', `${API_BASE_URL}/api/v1/analyze`);
    request.timeout = 120_000;

    const handleAbort = () => request.abort();
    const cleanup = () => signal.removeEventListener('abort', handleAbort);
    signal.addEventListener('abort', handleAbort, { once: true });

    request.upload.onprogress = (event) => {
      if (event.lengthComputable) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    };

    request.onload = () => {
      cleanup();
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(getHttpErrorMessage(request.status)));
        return;
      }

      try {
        const body: unknown = JSON.parse(request.responseText);
        if (!isRecord(body) || typeof body.job_id !== 'string' || !body.job_id) {
          throw new Error('Идентификатор задания отсутствует');
        }
        resolve(body.job_id);
      } catch {
        reject(new Error('API не вернул идентификатор задания.'));
      }
    };

    request.onerror = () => {
      cleanup();
      reject(new Error('Нет связи с API. Проверьте сеть и настройки CORS.'));
    };
    request.ontimeout = () => {
      cleanup();
      reject(new Error('Время отправки истекло. Попробуйте ещё раз.'));
    };
    request.onabort = () => {
      cleanup();
      reject(new DOMException('Отменено', 'AbortError'));
    };

    const form = new FormData();
    form.append('file', file);
    request.send(form);
  });
}

export async function getJob(jobId: string, signal: AbortSignal): Promise<AnalysisJob> {
  let response: Response;

  try {
    response = await fetch(`${API_BASE_URL}/api/v1/jobs/${encodeURIComponent(jobId)}`, {
      signal: AbortSignal.any([signal, AbortSignal.timeout(15_000)]),
    });
  } catch (error) {
    if (signal.aborted) throw error;
    const message =
      error instanceof DOMException && error.name === 'TimeoutError'
        ? 'API не ответил за 15 секунд. Повторите попытку.'
        : 'Нет связи с API. Проверьте сеть и настройки CORS.';
    throw new Error(message);
  }

  if (!response.ok) {
    throw new Error(getHttpErrorMessage(response.status));
  }

  return parseJob(await response.json());
}
