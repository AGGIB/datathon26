import { useEffect, useRef } from 'react';

interface EvidenceDialogProps {
  imageUrl: string;
  onClose: () => void;
}

export function EvidenceDialog({ imageUrl, onClose }: EvidenceDialogProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);

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
      <img src={imageUrl} alt="Увеличенный кадр-доказательство" />
    </dialog>
  );
}
