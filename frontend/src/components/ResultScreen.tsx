import { useEffect, useRef, useState } from 'react';
import { IS_DEMO } from '../config';
import { formatTime } from '../lib/utils';
import type { AnalysisResult } from '../types';
import { EvidenceDialog } from './EvidenceDialog';
import { ViolationCard } from './ViolationCard';

interface ResultScreenProps {
  file: File;
  result: AnalysisResult;
  onReset: () => void;
}

export function ResultScreen({ file, result, onReset }: ResultScreenProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [hasCodecError, setHasCodecError] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string>();
  const hasViolations = result.violations.length > 0;
  const timelineDuration =
    duration || Math.max(60, ...result.violations.map((v) => v.timecode_sec));

  useEffect(() => {
    const objectUrl = URL.createObjectURL(file);
    setVideoUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  function seekTo(seconds: number) {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = Math.max(0, Math.min(seconds, video.duration || seconds));
    video.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }

  return (
    <>
      <section className={`verdict ${hasViolations ? 'warning' : 'clear'}`} aria-live="polite">
        <div>
          <p className="eyebrow">{IS_DEMO ? 'Пример результата' : 'Проверка завершена'}</p>
          <h2>{result.verdict}</h2>
          <p>
            {hasViolations
              ? `Эпизодов для проверки: ${result.violations.length}`
              : 'Визуальных признаков нарушений не найдено.'}
          </p>
        </div>
        <button className="secondary" onClick={onReset}>
          Новая проверка
        </button>
      </section>

      <div className="result-layout">
        <section className="video-panel">
          <div className="panel-heading">
            <h2>Исходная запись</h2>
            <span>{file.name}</span>
          </div>
          <video
            ref={videoRef}
            src={videoUrl || undefined}
            controls
            preload="metadata"
            onLoadedMetadata={(event) => setDuration(event.currentTarget.duration)}
            onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
            onError={() => setHasCodecError(true)}
          />
          {hasCodecError && (
            <p className="error">
              Браузер не воспроизводит кодек этого видео. Используйте совместимый браузер или
              подключите MP4-превью от бэкенда.
            </p>
          )}
          <div className="time-label">
            <span>{formatTime(currentTime)}</span>
            <span>{duration ? formatTime(duration) : '—'}</span>
          </div>
          <div className="timeline" aria-label="Метки событий">
            <div
              className="playhead"
              style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%` }}
            />
            {result.violations.map((violation, index) => (
              <button
                key={index}
                title={`${violation.type}, ${formatTime(violation.timecode_sec)}`}
                aria-label={`Перейти к эпизоду ${index + 1}, ${formatTime(violation.timecode_sec)}`}
                style={{
                  left: `${Math.min(98, (violation.timecode_sec / timelineDuration) * 100)}%`,
                }}
                onClick={() => seekTo(violation.timecode_sec)}
              />
            ))}
          </div>
          <p className="caption">Нажмите метку или таймкод, чтобы перейти к эпизоду.</p>
        </section>

        <section className="findings">
          <div className="panel-heading">
            <h2>Эпизоды</h2>
            <span>{String(result.violations.length).padStart(2, '0')}</span>
          </div>
          {!hasViolations && (
            <div className="panel">
              <h3>Проверка завершена</h3>
              <p>
                Посмотрите исходную запись при необходимости. Вывод сервиса носит вспомогательный
                характер.
              </p>
            </div>
          )}
          {result.violations.map((violation, index) => (
            <ViolationCard
              key={index}
              violation={violation}
              index={index}
              onSeek={seekTo}
              onOpenImage={setSelectedImage}
            />
          ))}
        </section>
      </div>

      {selectedImage && (
        <EvidenceDialog imageUrl={selectedImage} onClose={() => setSelectedImage(undefined)} />
      )}
    </>
  );
}
