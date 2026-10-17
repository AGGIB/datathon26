import { ParticleBackground } from './components/ParticleBackground';
import { ResultScreen } from './components/ResultScreen';
import { UploadScreen } from './components/UploadScreen';
import { IS_DEMO } from './config';
import { useAnalysis } from './hooks/useAnalysis';
import './motion.css';

const STEPS = ['Загрузка записи', 'Обработка', 'Результат'];

export default function App() {
  const analysis = useAnalysis();
  const currentStep = analysis.phase === 'done' ? 2 : analysis.isBusy ? 1 : 0;

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
          <span className="environment">{IS_DEMO ? 'Локальное демо' : 'API настроен'}</span>
        </div>
      </header>

      <main id="workspace">
        <div className="workspace-heading">
          <div>
            <h1>Проверка кассовой записи</h1>
            <p className="lead">Загрузите видео, чтобы проверить события и их основания.</p>
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
            <strong>Демонстрация интерфейса.</strong> Видео остаётся на устройстве. Результаты
            вымышлены и показывают возможности просмотра.
          </aside>
        )}

        {analysis.error && (
          <div role="alert" className="error">
            {analysis.error}
          </div>
        )}

        {analysis.phase === 'upload' && (
          <UploadScreen
            file={analysis.file}
            apiKey={analysis.apiKey}
            showClearDemo={analysis.showClearDemo}
            onSelectFile={analysis.selectFile}
            onApiKeyChange={analysis.setApiKey}
            onDemoScenarioChange={analysis.setShowClearDemo}
            onStart={analysis.startAnalysis}
          />
        )}

        {analysis.isBusy && (
          <section className="processing panel" aria-live="polite">
            <div className="spinner" aria-hidden="true" />
            <p className="eyebrow">
              {analysis.phase === 'sending' ? 'Передача файла' : 'Проверка записи'}
            </p>
            <h2>
              {analysis.phase === 'sending'
                ? `Загружаем видео · ${analysis.uploadProgress}%`
                : analysis.stage}
            </h2>
            <p>Оставьте вкладку открытой. Результат появится здесь автоматически.</p>
            {analysis.phase === 'sending' && (
              <progress aria-label="Прогресс загрузки" max={100} value={analysis.uploadProgress} />
            )}
            {analysis.phase === 'processing' && analysis.processingProgress !== undefined && (
              <>
                <progress
                  aria-label="Прогресс обработки"
                  max={1}
                  value={analysis.processingProgress}
                />
                <p>{Math.round(analysis.processingProgress * 100)}%</p>
              </>
            )}
            <button className="secondary" onClick={analysis.reset}>
              Отменить ожидание
            </button>
          </section>
        )}

        {analysis.phase === 'done' && analysis.result && analysis.file && (
          <ResultScreen file={analysis.file} result={analysis.result} onReset={analysis.reset} />
        )}

        <footer>
          <span>Подсказка для проверяющего, не заключение</span>
          <span>Командный прототип · 2026</span>
        </footer>
      </main>
    </>
  );
}
