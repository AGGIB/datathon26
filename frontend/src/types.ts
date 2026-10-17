export interface Rule {
  document: string;
  clause?: string;
  quote?: string;
  url?: string;
}

export interface Violation {
  type: string;
  timecode_sec: number;
  description: string;
  evidence_frames: string[];
  rule?: Rule;
  confidence?: number;
  t_start?: number;
  t_end?: number;
}

export interface AnalysisResult {
  verdict: string;
  violations: Violation[];
}

export type AnalysisJob =
  | { status: 'processing'; stage?: string; progress?: number }
  | { status: 'done'; result: AnalysisResult }
  | { status: 'failed'; error: string };

export type AnalysisPhase = 'upload' | 'sending' | 'processing' | 'done';
