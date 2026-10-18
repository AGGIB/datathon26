import { formatTime, safeImageUrl, safeUrl } from '../lib/utils';
import type { Violation } from '../types';

interface ViolationCardProps {
  violation: Violation;
  index: number;
  onSeek: (seconds: number) => void;
  onOpenImage: (url: string) => void;
}

export function ViolationCard({ violation, index, onSeek, onOpenImage }: ViolationCardProps) {
  const { rule } = violation;
  const sourceUrl = rule?.url ? safeUrl(rule.url) : undefined;

  return (
    <article className="finding">
      <div className="finding-top">
        <span>Эпизод {String(index + 1).padStart(2, '0')}</span>
        <button className="timecode" onClick={() => onSeek(violation.timecode_sec)}>
          ▶ {formatTime(violation.timecode_sec)}
        </button>
      </div>
      <h3>{violation.type}</h3>
      <p>{violation.description}</p>

      <div className="frames">
        {violation.evidence_frames.slice(0, 3).map((frame, frameIndex) => {
          const imageUrl = safeImageUrl(frame);
          if (!imageUrl) return null;

          return (
            <button
              className="frame"
              key={frameIndex}
              onClick={() => onOpenImage(imageUrl)}
              aria-label={`Увеличить кадр ${frameIndex + 1}`}
            >
              <img
                src={imageUrl}
                alt={`Доказательство ${frameIndex + 1}, ${formatTime(violation.timecode_sec)}`}
                loading="lazy"
              />
            </button>
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
