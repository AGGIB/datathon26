import { formatEvidenceTime, formatTime, safeImageUrl, safeUrl } from '../lib/utils';
import type { EvidenceFrame, Violation } from '../types';

interface ViolationCardProps {
  violation: Violation;
  index: number;
  active: boolean;
  onActivate: () => void;
  canSeekTo: (seconds: number) => boolean;
  onSeek: (seconds: number) => void;
  onOpenImage: (frame: EvidenceFrame) => void;
}

export function ViolationCard({
  violation,
  index,
  active,
  onActivate,
  canSeekTo,
  onSeek,
  onOpenImage,
}: ViolationCardProps) {
  const { rule } = violation;
  const sourceUrl = rule?.url ? safeUrl(rule.url) : undefined;

  return (
    <article
      className={`finding ${active ? 'active' : ''}`}
      aria-labelledby={`title-${violation.id}`}
      onFocusCapture={onActivate}
      onClick={onActivate}
    >
      <div className="finding-top">
        <span>Эпизод {String(index + 1).padStart(2, '0')}</span>
        <button
          className="timecode"
          disabled={!canSeekTo(violation.timecode_sec)}
          onClick={() => onSeek(violation.timecode_sec)}
        >
          ▶ {formatTime(violation.timecode_sec)}
        </button>
      </div>
      <h3 id={`title-${violation.id}`}>{violation.type}</h3>
      <p>{violation.description}</p>

      <div className="frames">
        {violation.evidence_frames.slice(0, 3).map((frame, frameIndex) => {
          const imageUrl = safeImageUrl(frame.image_url);
          if (!imageUrl) return null;
          const frameTime = formatEvidenceTime(frame.timestamp_sec);

          return (
            <figure className="evidence-item" key={frameIndex}>
              <button
                className="frame"
                onClick={() => onOpenImage({ ...frame, image_url: imageUrl })}
                aria-label={`Увеличить кадр ${frameIndex + 1}, ${frameTime}`}
              >
                <img
                  src={imageUrl}
                  alt={`Доказательство ${frameIndex + 1}, ${frameTime}`}
                  loading="lazy"
                />
              </button>
              <figcaption>
                <button
                  className="timecode"
                  disabled={!canSeekTo(frame.timestamp_sec)}
                  onClick={() => onSeek(frame.timestamp_sec)}
                  aria-label={`Перейти к кадру ${frameIndex + 1}, ${frameTime}`}
                >
                  ▶ {frameTime}
                </button>
              </figcaption>
            </figure>
          );
        })}
      </div>
      {violation.evidence_frames.length === 0 && (
        <p className="caption">Кадры-доказательства не переданы.</p>
      )}

      <div className="rule">
        <b>Основание</b>
        {rule?.document ? (
          <>
            <p>
              {rule.document}
              {rule.clause ? ` · ${rule.clause}` : ''}
            </p>
            {rule.quote && <blockquote>{rule.quote}</blockquote>}
            {sourceUrl && (
              <a href={sourceUrl} target="_blank" rel="noopener noreferrer">
                Открыть источник
              </a>
            )}
          </>
        ) : (
          <p>Нормативное основание не передано. Требуется проверка специалистом.</p>
        )}
      </div>
    </article>
  );
}
