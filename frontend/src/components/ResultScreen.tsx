import { useEffect, useRef, useState } from 'react';
import { useObjectUrl } from '../hooks/useObjectUrl';
import { formatEvidenceTime, formatTime } from '../lib/utils';
import type { AnalysisResult, EvidenceFrame } from '../types';
import { EvidenceDialog } from './EvidenceDialog';
import { ViolationCard } from './ViolationCard';

interface ResultScreenProps {
  file: File;
  result: AnalysisResult;
  onReset: () => void;
}

export function ResultScreen({ file, result, onReset }: ResultScreenProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const videoUrl = useObjectUrl(file);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [hasCodecError, setHasCodecError] = useState(false);
  const [selectedImage, setSelectedImage] = useState<EvidenceFrame>();
  const [activeIndex, setActiveIndex] = useState(0);
  const [seekNotice, setSeekNotice] = useState('');
  const hasViolations = result.violations.length > 0;
  const active = result.violations[activeIndex];
  const timelineDuration =
    duration || Math.max(60, ...result.violations.map((v) => v.timecode_sec));
  const markers = [...new Set(result.violations.map((episode) => episode.timecode_sec))].map(
    (time) => ({
      time,
      indexes: result.violations.flatMap((episode, index) =>
        episode.timecode_sec === time ? [index] : [],
      ),
    }),
  );

  useEffect(() => {
    headingRef.current?.focus();
  }, []);

  function canSeekTo(seconds: number) {
    return (
      Number.isFinite(seconds) &&
      seconds >= 0 &&
      duration > 0 &&
      Number.isFinite(duration) &&
      seconds <= duration &&
      !hasCodecError
    );
  }

  function seekTo(seconds: number, focusVideo = true) {
    const video = videoRef.current;
    if (!video || !canSeekTo(seconds)) {
      setSeekNotice(
        'Переход недоступен: время должно находиться в пределах воспроизводимой записи.',
      );
      return;
    }
    video.currentTime = seconds;
    setCurrentTime(seconds);
    setSeekNotice('');
    // Run after a modal closes so its native focus restoration cannot undo the seek.
    window.requestAnimationFrame(() => {
      if (focusVideo) video.focus({ preventScroll: true });
      video.scrollIntoView({
        behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',
        block: 'nearest',
      });
    });
  }

  function selectEpisode(index: number) {
    const episode = result.violations[index];
    if (!episode) return;
    setActiveIndex(index);
    if (canSeekTo(episode.timecode_sec)) seekTo(episode.timecode_sec, false);
  }

  return (
    <>
      <section className={`verdict ${hasViolations ? 'warning' : 'clear'}`}>
        <span className="verdict-mark" aria-hidden="true">
          {hasViolations ? '!' : '✓'}
        </span>
        <div className="verdict-copy">
          <h2 ref={headingRef} tabIndex={-1}>
            {result.verdict}
          </h2>
          <p>Итоговую оценку даёт проверяющий.</p>
        </div>
        <div
          className="verdict-total"
          aria-label={`Количество эпизодов для проверки: ${result.violations.length}`}
        >
          <strong>{String(result.violations.length).padStart(2, '0')}</strong>
          <span>эпизодов</span>
        </div>
        <button className="secondary" onClick={onReset}>
          Новая проверка
        </button>
      </section>

      <div className="result-layout">
        <section className="video-panel" aria-labelledby="source-video-title">
          <div className="panel-heading">
            <h2 id="source-video-title">Исходная запись</h2>
            <span title={file.name}>{file.name}</span>
          </div>
          <video
            ref={videoRef}
            key={videoUrl}
            src={videoUrl}
            controls
            playsInline
            preload="metadata"
            aria-label="Исходная запись с переходом к эпизодам"
            onLoadedMetadata={(event) => {
              const next = event.currentTarget.duration;
              setDuration(Number.isFinite(next) && next > 0 ? next : 0);
            }}
            onTimeUpdate={(event) => {
              const time = event.currentTarget.currentTime;
              if (Number.isFinite(time)) setCurrentTime(time);
            }}
            onError={() => setHasCodecError(true)}
          />
          {hasCodecError && (
            <p className="error">Видео не воспроизводится. Описания и кадры эпизодов доступны.</p>
          )}
          <div className="time-label">
            <span>{formatTime(currentTime)}</span>
            <span>{duration ? formatTime(duration) : '—'}</span>
          </div>
          {hasViolations && (
            <>
              <div className="timeline" aria-label="Метки эпизодов">
                <div
                  className="playhead"
                  style={{
                    width: `${duration ? Math.min(100, (currentTime / duration) * 100) : 0}%`,
                  }}
                />
                {markers.map(({ time, indexes }) => (
                  <button
                    key={time}
                    className={indexes.includes(activeIndex) ? 'active' : ''}
                    title={`${indexes.length > 1 ? `Эпизоды ${indexes.map((i) => i + 1).join(', ')}` : `Эпизод ${indexes[0] + 1}`} · ${formatEvidenceTime(time)}`}
                    aria-label={`Выбрать ${indexes.length > 1 ? 'эпизоды' : 'эпизод'} ${indexes.map((i) => i + 1).join(', ')}, ${formatEvidenceTime(time)}`}
                    style={{
                      left: `${Math.max(2, Math.min(98, (time / timelineDuration) * 100))}%`,
                    }}
                    onClick={() =>
                      selectEpisode(
                        indexes.includes(activeIndex)
                          ? indexes[(indexes.indexOf(activeIndex) + 1) % indexes.length]
                          : indexes[0],
                      )
                    }
                  >
                    <span>{indexes.length > 1 ? indexes.length : ''}</span>
                  </button>
                ))}
              </div>
              <div className="episode-navigation">
                <button
                  className="secondary"
                  aria-label="Предыдущий эпизод"
                  disabled={activeIndex === 0}
                  onClick={() => selectEpisode(activeIndex - 1)}
                >
                  ←
                </button>
                <label htmlFor="active-episode">
                  <span className="sr-only">Выбрать эпизод</span>
                  <select
                    id="active-episode"
                    value={active?.id}
                    onChange={(event) =>
                      selectEpisode(
                        result.violations.findIndex((item) => item.id === event.target.value),
                      )
                    }
                  >
                    {result.violations.map((item, index) => (
                      <option key={item.id} value={item.id}>
                        Эпизод {index + 1} из {result.violations.length} ·{' '}
                        {formatEvidenceTime(item.timecode_sec)}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  className="secondary"
                  aria-label="Следующий эпизод"
                  disabled={activeIndex === result.violations.length - 1}
                  onClick={() => selectEpisode(activeIndex + 1)}
                >
                  →
                </button>
              </div>
              <p className="active-episode-summary" role="status">
                Эпизод {activeIndex + 1} из {result.violations.length} · {active?.type}
              </p>
            </>
          )}
          {seekNotice && (
            <p className="caption" role="status">
              {seekNotice}
            </p>
          )}
          {!hasCodecError && !duration && <p className="caption">Загрузка записи…</p>}
        </section>

        <section className="findings" aria-labelledby="findings-title">
          <div className="panel-heading">
            <h2 id="findings-title">Эпизоды</h2>
            <span>{String(result.violations.length).padStart(2, '0')}</span>
          </div>
          {!hasViolations && (
            <div className="panel empty-findings">
              <span aria-hidden="true">✓</span>
              <h3>Нет эпизодов для проверки</h3>
            </div>
          )}
          {result.violations.map((violation, index) => (
            <ViolationCard
              key={violation.id}
              violation={violation}
              index={index}
              active={activeIndex === index}
              onActivate={() => setActiveIndex(index)}
              canSeekTo={canSeekTo}
              onSeek={(seconds) => {
                setActiveIndex(index);
                seekTo(seconds);
              }}
              onOpenImage={(frame) => {
                setActiveIndex(index);
                setSelectedImage(frame);
              }}
            />
          ))}
        </section>
      </div>
      {selectedImage && (
        <EvidenceDialog
          frame={selectedImage}
          canSeek={canSeekTo(selectedImage.timestamp_sec)}
          onSeek={seekTo}
          onClose={() => setSelectedImage(undefined)}
        />
      )}
    </>
  );
}
