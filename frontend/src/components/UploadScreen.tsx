import { useEffect, useRef, useState } from 'react';
import { IS_DEMO, getApiKey, setApiKey } from '../config';

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
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState('');
  const [previewFailed, setPreviewFailed] = useState(false);
  const [apiKey, setApiKeyVal] = useState(() => getApiKey());

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

        <div className={`file-details ${file ? 'selected' : ''}`} aria-live="polite">
          <div className="file-summary">
            <span className="file-status-icon" aria-hidden="true">
              {file ? '✓' : '＋'}
            </span>
            <div>
              <span className="field-caption">
                {file ? 'Видео готово к проверке' : 'Видео не выбрано'}
              </span>
              <p>{file ? file.name : 'Выберите файл, чтобы начать проверку'}</p>
            </div>
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
              Ключ доступа (X-API-Key)
              <input
                type="password"
                name="apiKey"
                value={apiKey}
                placeholder="Введите ключ доступа"
                onChange={(event) => {
                  const val = event.target.value;
                  setApiKeyVal(val);
                  setApiKey(val);
                }}
              />
              <small>Введите ключ, полученный от команды.</small>
            </label>
          )}
          <button
            className="start-button"
            disabled={!file || (!IS_DEMO && !apiKey.trim())}
            onClick={onStart}
          >
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
