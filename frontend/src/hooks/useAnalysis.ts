import { useEffect, useRef, useState } from 'react';
import { getJob, uploadVideo } from '../api';
import { ANALYSIS_TIMEOUT_MS, IS_DEMO, MAX_FILE_SIZE, POLL_INTERVAL_MS } from '../config';
import { DEMO_CLEAR_RESULT, DEMO_RESULT } from '../fixtures';
import { createAnalysisRunGuard, validateVideoFile } from '../lib/analysis-state';
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
  const runGuardRef = useRef(createAnalysisRunGuard());

  useEffect(
    () => () => {
      runGuardRef.current.invalidate();
      controllerRef.current?.abort();
      controllerRef.current = null;
    },
    [],
  );

  function invalidateRun() {
    // Invalidate first: abort may synchronously trigger transport callbacks.
    runGuardRef.current.invalidate();
    const controller = controllerRef.current;
    controllerRef.current = null;
    controller?.abort();
  }

  function clearRunState() {
    setPhase('upload');
    setResult(undefined);
    setStage('');
    setError('');
    setUploadProgress(0);
    setProcessingProgress(undefined);
  }

  function selectFile(selectedFile?: File) {
    invalidateRun();
    clearRunState();
    const selectionError = validateVideoFile(selectedFile, MAX_FILE_SIZE);
    // A rejected replacement must not leave an older file ready to submit.
    setFile(selectionError ? undefined : selectedFile);
    setError(selectionError);
  }

  function cancel() {
    invalidateRun();
    clearRunState();
  }

  function reset() {
    cancel();
    setFile(undefined);
  }

  async function startAnalysis() {
    if (!file) return;

    invalidateRun();
    const run = runGuardRef.current.begin();
    const controller = new AbortController();
    controllerRef.current = controller;
    const { signal } = controller;
    const isCurrent = () => run.isCurrent() && !signal.aborted;
    const commit = (update: () => void) => {
      if (isCurrent()) run.commit(update);
    };
    setError('');
    setResult(undefined);
    setStage('');
    setPhase('sending');
    setUploadProgress(0);
    setProcessingProgress(undefined);

    try {
      if (IS_DEMO) {
        setUploadProgress(100);
        setPhase('processing');
        setStage('Демонстрация интерфейса');
        await delay(1_600, signal);
        commit(() => {
          setResult(showClearDemo ? DEMO_CLEAR_RESULT : DEMO_RESULT);
          setPhase('done');
        });
        return;
      }

      const jobId = await uploadVideo(
        file,
        (progress) => commit(() => setUploadProgress(progress)),
        signal,
      );
      if (!isCurrent()) return;
      setPhase('processing');
      setStage('Видео в очереди обработки');
      const deadline = Date.now() + ANALYSIS_TIMEOUT_MS;

      while (isCurrent()) {
        if (Date.now() > deadline) {
          throw new Error(
            'Ожидание превысило 5 минут. Повторите попытку или уточните состояние API у команды.',
          );
        }

        const job = await getJob(jobId, signal);
        if (!isCurrent()) return;
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
      commit(() => {
        setError(error instanceof Error ? error.message : 'Не удалось связаться с сервисом.');
        setPhase('upload');
      });
    } finally {
      if (run.isCurrent()) {
        runGuardRef.current.invalidate();
        controllerRef.current = null;
      }
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
    cancel,
    reset,
    startAnalysis,
    isBusy: phase === 'sending' || phase === 'processing',
  };
}
