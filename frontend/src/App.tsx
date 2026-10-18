import { useEffect, useRef, useState } from 'react';
import { ParticleBackground } from './components/ParticleBackground';
import { ResultScreen } from './components/ResultScreen';
import { UploadScreen } from './components/UploadScreen';
import { IS_DEMO } from './config';
import { useAnalysis } from './hooks/useAnalysis';
import { formatTime } from './lib/utils';
import './motion.css';

const STEPS = ['Загрузка', 'Обработка', 'Результат'];

export default function App() {
  const analysis = useAnalysis();
  const [notice, setNotice] = useState('');
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const errorRef = useRef<HTMLDivElement>(null);
  const currentStep = analysis.phase === 'done' ? 2 : analysis.phase === 'processing' ? 1 : 0;

  useEffect(() => {
    if (!analysis.isBusy) return;
    const startedAt = performance.now();
    setElapsedSeconds(0);
    const timer = window.setInterval(
      () => setElapsedSeconds(Math.floor((performance.now() - startedAt) / 1000)),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [analysis.isBusy]);

  useEffect(() => {
    if (analysis.error) errorRef.current?.focus();
  }, [analysis.error]);

  function cancelWaiting() {
    analysis.cancel();
    setNotice(
      IS_DEMO
        ? 'Демонстрация отменена. Можно изменить сценарий или выбрать другую запись.'
        : 'Ожидание остановлено. Файл сохранён для повторной проверки. Уже созданная задача на сервере продолжает выполняться.',
    );
  }

  return (
    <>
      <ParticleBackground />
      <a className="skip-link" href="#workspace">
        Перейти к проверке
      </a>
      <header className="app-header">
        <a className="brand" href="/" aria-label="Кассовый контроль — главная">
          <img className="forte-logo" src="/forte-logo.svg" width="121" height="36" alt="Forte" />
          <span className="brand-service">Кассовый контроль</span>
        </a>
        <div className="header-meta">
          <span className="project-label">Datathon 26 · Прототип</span>
          <span className="environment">{IS_DEMO ? 'Локальное демо' : 'Рабочий режим'}</span>
        </div>
      </header>

      <main id="workspace" tabIndex={-1}>
        <div className="workspace-heading">
          <div>
            <h1>Проверка кассовой записи</h1>
            <p className="lead">Видео, отмеченные эпизоды и основания — в одном окне.</p>
          </div>
        </div>

        <nav className="steps" aria-label="Этапы проверки">
          <ol>
            {STEPS.map((step, index) => (
              <li
                key={step}
                aria-current={currentStep === index ? 'step' : undefined}
                className={
                  currentStep === index ? 'current' : currentStep > index ? 'complete' : ''
                }
              >
                <span className="step-number" aria-hidden="true">
                  {currentStep > index ? '✓' : index + 1}
                </span>
                {step}
              </li>
            ))}
          </ol>
        </nav>

        {IS_DEMO && (
          <aside className="demo-banner">
            <strong>Демо.</strong> Видео остаётся на устройстве. Результаты и иллюстрации вымышлены.
          </aside>
        )}

        {notice && (
          <p className="notice" role="status">
            {notice}
          </p>
        )}
        {analysis.error && (
          <div ref={errorRef} tabIndex={-1} role="alert" className="error">
            <strong>Проверьте данные и попробуйте ещё раз.</strong>
            <span>{analysis.error}</span>
          </div>
        )}

        {analysis.phase === 'upload' && (
          <UploadScreen
            file={analysis.file}
            showClearDemo={analysis.showClearDemo}
            onSelectFile={(file) => {
              setNotice('');
              analysis.selectFile(file);
            }}
            onDemoScenarioChange={analysis.setShowClearDemo}
            onStart={() => {
              setNotice('');
              analysis.startAnalysis();
            }}
          />
        )}

        {analysis.isBusy && (
          <section className="processing panel" aria-label="Статус проверки" aria-busy="true">
            <div className="spinner" aria-hidden="true" />
            <p className="eyebrow">
              {IS_DEMO
                ? 'Демонстрация интерфейса'
                : analysis.phase === 'sending'
                  ? 'Передача файла'
                  : 'Проверка записи'}
            </p>
            <div role="status" aria-live="polite">
              <h2>
                {analysis.phase === 'sending'
                  ? `Загружаем видео · ${analysis.uploadProgress}%`
                  : analysis.stage}
              </h2>
            </div>
            <p className="processing-file">{analysis.file?.name}</p>
            <div className="elapsed-time" aria-live="off">
              Прошло <strong>{formatTime(elapsedSeconds)}</strong>
            </div>
            {analysis.phase === 'sending' ? (
              <progress aria-label="Прогресс загрузки" max={100} value={analysis.uploadProgress} />
            ) : (
              <progress
                aria-label="Прогресс обработки"
                max={1}
                value={analysis.processingProgress}
              />
            )}
            {analysis.phase === 'processing' && analysis.processingProgress !== undefined && (
              <p className="processing-percent">{Math.round(analysis.processingProgress * 100)}%</p>
            )}
            <p>Результат появится здесь. Оставьте вкладку открытой.</p>
            {!IS_DEMO && (
              <p className="caption">
                Время зависит от записи и загрузки сервиса. Лимит ожидания в интерфейсе — 5 минут.
              </p>
            )}
            <button className="secondary" onClick={cancelWaiting}>
              Отменить ожидание
            </button>
            <p className="cancel-note">
              {IS_DEMO
                ? 'Выбранный файл останется на устройстве.'
                : 'Отменяется только ожидание в браузере. Задача на сервере продолжит работу.'}
            </p>
          </section>
        )}

        {analysis.phase === 'done' && analysis.result && analysis.file && (
          <ResultScreen
            file={analysis.file}
            result={analysis.result}
            onReset={() => {
              setNotice('');
              analysis.reset();
            }}
          />
        )}

        <footer>
          <span>Подсказка для проверяющего, не заключение</span>
          <span>Командный прототип · 2026</span>
        </footer>
      </main>
    </>
  );
}
