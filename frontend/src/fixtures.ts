import type { AnalysisResult } from './types';

export const DEMO_RESULT: AnalysisResult = {
  verdict: 'Обнаружены признаки возможных нарушений',
  violations: [
    {
      type: 'Демонстрационный эпизод: открытый сейф',
      timecode_sec: 22,
      description:
        'Пример карточки интерфейса. В реальном анализе здесь будет описание только наблюдаемых действий из ответа пайплайна. Эта карточка не является результатом анализа загруженного видео.',
      evidence_frames: [],
    },
    {
      type: 'Демонстрационный эпизод: доступ в кассу',
      timecode_sec: 48,
      description:
        'Второй пример показывает навигацию между эпизодами. Нормативное основание и кадры появятся после подключения API команды.',
      evidence_frames: [],
    },
  ],
};

export const DEMO_CLEAR_RESULT: AnalysisResult = {
  verdict: 'Нарушение не обнаружено',
  violations: [],
};
