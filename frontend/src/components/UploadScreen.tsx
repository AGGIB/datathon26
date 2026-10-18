import { useEffect, useRef, useState } from 'react';
import { IS_DEMO, getApiKey, setApiKey } from '../config';
import { useObjectUrl } from '../hooks/useObjectUrl';

interface UploadScreenProps {
  file?: File;
  showClearDemo: boolean;
  onSelectFile: (file?: File) => void;
  onDemoScenarioChange: (value: boolean) => void;
  onStart: () => void;
}

export function UploadScreen({
  file,
  showClearDemo,
  onSelectFile,
  onDemoScenarioChange,
  onStart,
}: UploadScreenProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const chooseRef = useRef<HTMLButtonElement>(null);
  const dragDepth = useRef(0);
  const [isDragging, setIsDragging] = useState(false);
  const previewUrl = useObjectUrl(file);
  const [failedPreview, setFailedPreview] = useState<File>();
  const [apiKey, setApiKeyVal] = useState(() => getApiKey());
  const [showApiKey, setShowApiKey] = useState(false);
  const previewFailed = Boolean(file && failedPreview === file);

  useEffect(() => {
    dragDepth.current = 0;
    setIsDragging(false);
  }, [file]);

  function removeFile() {
    onSelectFile(undefined);
    window.requestAnimationFrame(() => chooseRef.current?.focus());
  }

  return (
    <section className="upload-layout" aria-label="Новая проверка">
      <div className="upload-workspace">
        <div className="workspace-section-title">
          <h2 id="upload-heading">Исходная запись</h2>
          <span id="upload-help">MOV, MP4 · до 200 МиБ</span>
        </div>

        <div
          className={`dropzone ${isDragging ? 'drag' : ''} ${file ? 'has-file' : ''}`}
          role="group"
          aria-labelledby="upload-heading"
          aria-describedby="upload-help"
          onDragEnter={(event) => {
            event.preventDefault();
            dragDepth.current++;
            setIsDragging(true);
          }}
          onDragOver={(event) => {
            event.preventDefault();
            event.dataTransfer.dropEffect = 'copy';
          }}
          onDragLeave={(event) => {
            event.preventDefault();
            dragDepth.current = Math.max(0, dragDepth.current - 1);
            if (dragDepth.current === 0) setIsDragging(false);
          }}
          onDrop={(event) => {
            event.preventDefault();
            dragDepth.current = 0;
            setIsDragging(false);
            onSelectFile(event.dataTransfer.files[0]);
          }}
        >
          {file && previewUrl && !previewFailed ? (
            <video
              key={previewUrl}
              src={previewUrl}
              controls
              playsInline
              preload="metadata"
              aria-label="Предпросмотр выбранной записи"
              onError={() => setFailedPreview(file)}
            />
          ) : (
            <div className="dropzone-content">
              <svg className="video-outline" viewBox="0 0 64 48" fill="none" aria-hidden="true">
                <rect x="2" y="2" width="44" height="44" rx="5" />
                <path d="m46 17 15-9v32l-15-9M17 24h14m-7-7v14" />
              </svg>
              <h3>{file ? 'Запись выбрана' : 'Добавьте видео для проверки'}</h3>
              <p>
                {previewFailed
                  ? 'Предпросмотр недоступен для этого кодека. Файл можно отправить на проверку.'
                  : 'Перетащите файл сюда или выберите на устройстве.'}
              </p>
              <button ref={chooseRef} onClick={() => inputRef.current?.click()}>
                {file ? 'Выбрать другой файл' : 'Выбрать видео'}
              </button>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            name="video"
            accept=".mov,.mp4,video/quicktime,video/mp4"
            hidden
            onChange={(event) => {
              const selected = event.target.files?.[0];
              if (selected) onSelectFile(selected);
              event.target.value = '';
            }}
          />
        </div>

        <div className={`file-details ${file ? 'selected' : ''}`} aria-live="polite">
          <div className="file-summary">
            <span className="file-status-icon" aria-hidden="true">
              {file ? '✓' : '＋'}
            </span>
            <div>
              <span className="field-caption">{file ? 'Файл выбран' : 'Видео не выбрано'}</span>
              <p>{file ? file.name : 'Начните с выбора записи'}</p>
            </div>
          </div>
          {file && (
            <div className="file-actions">
              <span>{(file.size / 1024 / 1024).toFixed(1)} МиБ</span>
              <button className="text-button" onClick={() => inputRef.current?.click()}>
                Заменить
              </button>
              <button className="text-button" onClick={removeFile}>
                Удалить
              </button>
            </div>
          )}
        </div>

        <div className="analysis-settings">
          {IS_DEMO && (
            <label>
              Пример результата
              <select
                name="demoScenario"
                value={showClearDemo ? 'clear' : 'events'}
                onChange={(event) => onDemoScenarioChange(event.target.value === 'clear')}
              >
                <option value="events">Два эпизода для проверки</option>
                <option value="clear">Нарушение не обнаружено</option>
              </select>
            </label>
          )}
          <div className="start-action">
            <button className="start-button" disabled={!file} onClick={onStart}>
              {IS_DEMO ? 'Открыть пример' : 'Начать проверку'}
            </button>
            {!file && <span className="caption">Сначала выберите видео</span>}
          </div>
        </div>
        {!IS_DEMO && (
          <details
            className="connection-settings"
            onToggle={(event) => {
              if (!event.currentTarget.open) setShowApiKey(false);
            }}
          >
            <summary>Настройки подключения</summary>
            <div className="connection-fields">
              <label htmlFor="api-key">Ключ доступа к сервису</label>
              <div className="api-key-field">
                <input
                  id="api-key"
                  type={showApiKey ? 'text' : 'password'}
                  name="apiKey"
                  value={apiKey}
                  autoComplete="off"
                  autoCapitalize="none"
                  spellCheck={false}
                  aria-describedby="api-key-help"
                  onChange={(event) => {
                    const value = event.target.value;
                    setApiKeyVal(value);
                    setApiKey(value);
                  }}
                />
                <button
                  className="secondary"
                  type="button"
                  aria-pressed={showApiKey}
                  onClick={() => setShowApiKey((shown) => !shown)}
                >
                  {showApiKey ? 'Скрыть' : 'Показать'}
                </button>
              </div>
              <p id="api-key-help" className="caption">
                Используется ключ интерфейса. Измените его, если команда выдала другой. Настройка
                сохраняется в этом браузере, если хранилище доступно.
              </p>
            </div>
          </details>
        )}
      </div>

      <aside className="review-guide" aria-labelledby="review-guide-title">
        <p className="guide-eyebrow">После проверки</p>
        <h2 id="review-guide-title">От записи к эпизоду</h2>
        <dl className="result-details">
          <div>
            <dt>
              <span aria-hidden="true">01</span> События и таймкоды
            </dt>
            <dd>Переходите к отмеченным моментам в исходной записи.</dd>
          </div>
          <div>
            <dt>
              <span aria-hidden="true">02</span> Кадры эпизода
            </dt>
            <dd>Сравнивайте кадры и открывайте их в увеличении.</dd>
          </div>
          <div>
            <dt>
              <span aria-hidden="true">03</span> Основания
            </dt>
            <dd>Читайте документ и источник, если они переданы сервисом.</dd>
          </div>
        </dl>
        <p className="review-note">
          Сервис помогает найти эпизоды. Итоговую оценку даёт проверяющий.
        </p>
      </aside>
    </section>
  );
}
