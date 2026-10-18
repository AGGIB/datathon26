export interface VideoFileLike {
  readonly name: string;
  readonly size: number;
}

export function validateVideoFile(file: VideoFileLike | undefined, maxBytes: number): string {
  if (!file) return '';
  if (!/\.(mov|mp4)$/i.test(file.name)) {
    return 'Выберите видео в формате MOV или MP4.';
  }
  if (!Number.isSafeInteger(file.size) || file.size <= 0 || file.size > maxBytes) {
    return 'Файл должен быть непустым и не больше 200 МБ.';
  }
  return '';
}

export interface AnalysisRun {
  isCurrent(): boolean;
  commit(update: () => void): boolean;
}

/** A late result, progress callback or error cannot write into a newer run. */
export function createAnalysisRunGuard() {
  let current: symbol | undefined;

  return {
    begin(): AnalysisRun {
      const token = Symbol('analysis-run');
      current = token;
      return {
        isCurrent: () => current === token,
        commit(update) {
          if (current !== token) return false;
          update();
          return true;
        },
      };
    },
    invalidate() {
      current = undefined;
    },
  };
}
