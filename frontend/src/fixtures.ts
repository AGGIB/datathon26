import type { AnalysisResult } from './types';

export const DEMO_RESULT: AnalysisResult = {
  verdict: 'Обнаружены признаки возможных нарушений',
  violations: [
    {
      id: 'demo-episode-1',
      type: 'Демонстрационный эпизод: открытый сейф',
      timecode_sec: 22,
      description:
        'Пример карточки интерфейса с синтетическими изображениями. В реальном анализе здесь будет описание только наблюдаемых действий из ответа пайплайна. Эта карточка не является результатом анализа загруженного видео.',
      evidence_frames: [
        { timestamp_sec: 18, image_url: '/demo/evidence-1.jpg' },
        { timestamp_sec: 22, image_url: '/demo/evidence-2.jpg' },
        { timestamp_sec: 27, image_url: '/demo/evidence-3.jpg' },
      ],
    },
    {
      id: 'demo-episode-2',
      type: 'Демонстрационный эпизод: доступ в кассу',
      timecode_sec: 42,
      description:
        'Второй вымышленный пример показывает навигацию между эпизодами. Синтетические изображения предназначены только для проверки интерфейса и не являются доказательствами нарушения.',
      evidence_frames: [
        { timestamp_sec: 36, image_url: '/demo/evidence-1.jpg' },
        { timestamp_sec: 42, image_url: '/demo/evidence-2.jpg' },
        { timestamp_sec: 47, image_url: '/demo/evidence-3.jpg' },
      ],
    },
  ],
};

export const DEMO_CLEAR_RESULT: AnalysisResult = {
  verdict: 'Нарушение не обнаружено',
  violations: [],
};
