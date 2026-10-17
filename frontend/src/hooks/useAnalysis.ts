import { useEffect, useRef, useState } from 'react';
import { getJob, uploadVideo } from '../api';
import { ANALYSIS_TIMEOUT_MS, IS_DEMO, MAX_FILE_SIZE, POLL_INTERVAL_MS } from '../config';
import { DEMO_CLEAR_RESULT, DEMO_RESULT } from '../fixtures';
import { delay } from '../lib/utils';
import type { AnalysisPhase, AnalysisResult } from '../types';

export function useAnalysis() {
  const [file, setFile] = useState<File>();
  const [phase, setPhase] = useState<AnalysisPhase>('upload');
  const [uploadProgress, setUploadProgress] = useState(0);
  const [stage, setStage] = useState('');
  const [processingProgress, setProcessingProgress] = useState<number>();
  const [error, setError] = useState('');
  const [result, setResult] = useState<AnalysisResult>();
  const [showClearDemo, setShowClearDemo] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => () => controllerRef.current?.abort(), []);

  function selectFile(selectedFile?: File) {
    if (!selectedFile) return;
    setError('');

    if (!/\.(mov|mp4)$/i.test(selectedFile.name)) {
      setError('Выберите видео в формате MOV или MP4.');
      return;
    }
    if (selectedFile.size === 0 || selectedFile.size > MAX_FILE_SIZE) {
      setError('Файл должен быть непустым и не больше 200 МБ.');
      return;
    }

    setFile(selectedFile);
    setResult(undefined);
  }

  function reset() {
    controllerRef.current?.abort();
    setPhase('upload');
    setResult(undefined);
    setError('');
    setUploadProgress(0);
    setProcessingProgress(undefined);
  }

  async function startAnalysis() {
    if (!file) return;

    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    const { signal } = controller;
    setError('');
    setPhase('sending');
    setUploadProgress(0);
    setProcessingProgress(undefined);

    try {
      if (IS_DEMO) {
        setUploadProgress(100);
        setPhase('processing');
        setStage('Демонстрация интерфейса');
        await delay(1_600, signal);
        setResult(showClearDemo ? DEMO_CLEAR_RESULT : DEMO_RESULT);
        setPhase('done');
        return;
      }

      const jobId = await uploadVideo(file, setUploadProgress, signal);
      signal.throwIfAborted();
      setPhase('processing');
      setStage('Видео в очереди обработки');
      const deadline = Date.now() + ANALYSIS_TIMEOUT_MS;

      while (!signal.aborted) {
        if (Date.now() > deadline) {
          throw new Error(
            'Ожидание превысило 5 минут. Повторите попытку или уточните состояние API у команды.',
          );
        }

        const job = await getJob(jobId, signal);
        signal.throwIfAborted();
        if (job.status === 'failed') {
          throw new Error(job.error || 'Не удалось обработать видео. Попробуйте другой файл.');
        }
        if (job.status === 'done') {
          setResult(job.result);
          setPhase('done');
          return;
        }

        setStage(job.stage || 'Анализ видео выполняется');
        setProcessingProgress(job.progress);
        await delay(POLL_INTERVAL_MS, signal);
      }
    } catch (error) {
      if (signal.aborted) return;
      setError(error instanceof Error ? error.message : 'Не удалось связаться с сервисом.');
      setPhase('upload');
    }
  }

  return {
    file,
    phase,
    uploadProgress,
    processingProgress,
    stage,
    error,
    result,
    showClearDemo,
    setShowClearDemo,
    selectFile,
    reset,
    startAnalysis,
    isBusy: phase === 'sending' || phase === 'processing',
  };
}
