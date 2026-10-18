import { useEffect, useState } from 'react';

/** A URL belongs to exactly one file and is released on replacement or unmount. */
export function useObjectUrl(file?: File): string | undefined {
  const [preview, setPreview] = useState<{ file: File; url: string }>();

  useEffect(() => {
    if (!file) {
      setPreview(undefined);
      return;
    }

    const url = URL.createObjectURL(file);
    setPreview({ file, url });
    return () => URL.revokeObjectURL(url);
  }, [file]);

  return preview?.file === file ? preview?.url : undefined;
}
