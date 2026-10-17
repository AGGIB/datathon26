import { useEffect, useRef, useState } from 'react';
import { IS_DEMO } from '../config';

interface UploadScreenProps {
  file?: File;
  apiKey: string;
  showClearDemo: boolean;
  onSelectFile: (file?: File) => void;
  onApiKeyChange: (value: string) => void;
  onDemoScenarioChange: (value: boolean) => void;
  onStart: () => void;
}

export function UploadScreen({
  file,
  apiKey,
  showClearDemo,
  onSelectFile,
  onApiKeyChange,
  onDemoScenarioChange,
  onStart,
}: UploadScreenProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState('');
  const [previewFailed, setPreviewFailed] = useState(false);

  useEffect(() => {
    setPreviewFailed(false);
    if (!file) return;
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return (
    <section className="upload-layout" aria-label="Новая проверка">
      <div className="upload-workspace">
        <div className="workspace-section-title">
          <h2>Исходная запись</h2>
          <span>MOV, MP4 / до 200 МБ</span>
        </div>

        <div
          className={`dropzone ${isDragging ? 'drag' : ''} ${file ? 'has-file' : ''}`}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setIsDragging(false);
            onSelectFile(event.dataTransfer.files[0]);
          }}
        >
          {file && previewUrl && !previewFailed ? (
            <video
              src={previewUrl}
              controls
              playsInline
              preload="metadata"
              aria-label="Предпросмотр выбранной записи"
              onError={() => setPreviewFailed(true)}
            />
          ) : (
            <div className="dropzone-content">
              <svg className="video-outline" viewBox="0 0 64 48" fill="none" aria-hidden="true">
                <rect x="2" y="2" width="44" height="44" rx="5" />
                <path d="m46 17 15-9v32l-15-9M17 24h14m-7-7v14" />
              </svg>
              <h3>{file ? 'Запись выбрана' : 'Перетащите видео в эту область'}</h3>
              <p>
                {previewFailed
                  ? 'Кодек не поддерживается браузером. Файл можно отправить на проверку.'
                  : 'Или выберите запись на устройстве.'}
              </p>
              <button className="secondary" onClick={() => inputRef.current?.click()}>
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
              onSelectFile(event.target.files?.[0]);
              event.target.value = '';
            }}
          />
        </div>

        <div className="file-details" aria-live="polite">
          <div>
            <span className="field-caption">{file ? 'Выбранный файл' : 'Файл не выбран'}</span>
            <p>{file ? file.name : 'Добавьте запись, чтобы продолжить'}</p>
          </div>
          {file && (
            <div className="file-actions">
              <span>{(file.size / 1024 / 1024).toFixed(1)} МБ</span>
              <button className="text-button" onClick={() => inputRef.current?.click()}>
                Заменить
              </button>
            </div>
          )}
        </div>

        <div className="analysis-settings">
          {IS_DEMO ? (
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
          ) : (
            <label>
              Ключ доступа
              <input
                name="apiKey"
                type="password"
                autoComplete="off"
                spellCheck={false}
                value={apiKey}
                onChange={(event) => onApiKeyChange(event.target.value)}
                placeholder="Введите UI API-ключ"
              />
              <small>Хранится в памяти этой вкладки.</small>
            </label>
          )}
          <button className="start-button" disabled={!file} onClick={onStart}>
            {IS_DEMO ? 'Открыть пример' : 'Начать проверку'}
          </button>
        </div>
      </div>

      <aside className="review-guide" aria-labelledby="review-guide-title">
        <h2 id="review-guide-title">Что будет в результате</h2>
        <p className="guide-intro">
          Всё необходимое, чтобы самостоятельно проверить каждый эпизод.
        </p>
        <dl className="result-details">
          <div>
            <dt>Событие и таймкод</dt>
            <dd>Описание того, что видно, и переход к моменту в записи.</dd>
          </div>
          <div>
            <dt>Кадры-доказательства</dt>
            <dd>До трёх кадров на эпизод. Детали можно рассмотреть в увеличении.</dd>
          </div>
          <div>
            <dt>Нормативное основание</dt>
            <dd>Документ, пункт и цитата со ссылкой на источник, если они переданы сервисом.</dd>
          </div>
        </dl>
        <div className="review-note">
          <h3>Решение остаётся за вами</h3>
          <p>Сервис отмечает признаки возможных нарушений. Итоговую оценку даёт проверяющий.</p>
        </div>
        <p className="processing-estimate">Ожидаемое время обработки — до 5 минут.</p>
      </aside>
    </section>
  );
}
