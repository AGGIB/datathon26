import { useEffect, useRef } from 'react';
import { formatEvidenceTime } from '../lib/utils';
import type { EvidenceFrame } from '../types';

interface EvidenceDialogProps {
  frame: EvidenceFrame;
  canSeek: boolean;
  onSeek: (seconds: number) => void;
  onClose: () => void;
}

export function EvidenceDialog({ frame, canSeek, onSeek, onClose }: EvidenceDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const frameTime = formatEvidenceTime(frame.timestamp_sec);

  useEffect(() => {
    const dialog = dialogRef.current;
    dialog?.showModal();
    return () => dialog?.close();
  }, []);

  return (
    <dialog
      ref={dialogRef}
      className="modal"
      aria-label="Кадр-доказательство"
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <button autoFocus onClick={onClose} aria-label="Закрыть увеличенный кадр">
        Закрыть ×
      </button>
      <figure className="evidence-dialog-content">
        <img src={frame.image_url} alt={`Увеличенный кадр-доказательство, ${frameTime}`} />
        <figcaption>
          <span>Кадр {frameTime}</span>
          <button
            disabled={!canSeek}
            onClick={() => {
              onSeek(frame.timestamp_sec);
              onClose();
            }}
          >
            Перейти к этому кадру
          </button>
        </figcaption>
      </figure>
    </dialog>
  );
}
