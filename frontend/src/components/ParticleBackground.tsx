import { useEffect, useRef } from 'react';

interface Particle {
  x: number;
  y: number;
  driftX: number;
  driftY: number;
  impulseX: number;
  impulseY: number;
  radius: number;
}

interface Ripple {
  x: number;
  y: number;
  startedAt: number;
}

/** Pointer listeners observe the page without intercepting its controls. */
export function ParticleBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;

    const surface = canvas;
    const brush = context;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    const finePointer = window.matchMedia('(pointer: fine)');
    const pointer = { x: -1000, y: -1000, active: false };
    let width = 0;
    let height = 0;
    let particles: Particle[] = [];
    let ripples: Ripple[] = [];
    let animationFrame = 0;
    let lastFrameTime = 0;

    function resize() {
      width = window.innerWidth;
      height = window.innerHeight;
      const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
      surface.width = Math.round(width * pixelRatio);
      surface.height = Math.round(height * pixelRatio);
      brush.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      const maxParticles = finePointer.matches ? 85 : 28;
      const count = Math.min(maxParticles, Math.round((width * height) / 16_000));
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        driftX: (Math.random() - 0.5) * 0.35,
        driftY: (Math.random() - 0.5) * 0.35,
        impulseX: 0,
        impulseY: 0,
        radius: Math.random() * 1.3 + 1,
      }));
      ripples = [];
      restart();
    }

    function drawLine(x1: number, y1: number, x2: number, y2: number, opacity: number) {
      brush.strokeStyle = `rgba(163, 21, 81, ${opacity})`;
      brush.lineWidth = 0.7;
      brush.beginPath();
      brush.moveTo(x1, y1);
      brush.lineTo(x2, y2);
      brush.stroke();
    }

    function draw(timestamp: number) {
      const frameScale = lastFrameTime ? Math.min((timestamp - lastFrameTime) / 16.67, 2) : 1;
      lastFrameTime = timestamp;
      brush.clearRect(0, 0, width, height);

      for (const particle of particles) {
        const deltaX = particle.x - pointer.x;
        const deltaY = particle.y - pointer.y;
        const distance = Math.hypot(deltaX, deltaY);

        if (!reducedMotion.matches) {
          if (pointer.active && distance > 0 && distance < 120) {
            const force = (1 - distance / 120) * 0.08 * frameScale;
            particle.impulseX += (deltaX / distance) * force;
            particle.impulseY += (deltaY / distance) * force;
          }
          particle.x += (particle.driftX + particle.impulseX) * frameScale;
          particle.y += (particle.driftY + particle.impulseY) * frameScale;
          const drag = Math.pow(0.96, frameScale);
          particle.impulseX *= drag;
          particle.impulseY *= drag;
          if (particle.x < -15) particle.x = width + 15;
          if (particle.x > width + 15) particle.x = -15;
          if (particle.y < -15) particle.y = height + 15;
          if (particle.y > height + 15) particle.y = -15;
        }

        const highlighted = pointer.active && distance < 190 && !reducedMotion.matches;
        brush.fillStyle = highlighted ? 'rgba(163, 21, 81, 0.7)' : 'rgba(166, 81, 119, 0.4)';
        brush.beginPath();
        brush.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
        brush.fill();
        if (highlighted) {
          drawLine(pointer.x, pointer.y, particle.x, particle.y, (1 - distance / 190) * 0.22);
        }
      }

      for (let first = 0; first < particles.length; first++) {
        for (let second = first + 1; second < particles.length; second++) {
          const a = particles[first];
          const b = particles[second];
          const distance = Math.hypot(a.x - b.x, a.y - b.y);
          if (distance < 145) drawLine(a.x, a.y, b.x, b.y, (1 - distance / 145) * 0.24);
        }
      }

      ripples = ripples.filter((ripple) => timestamp - ripple.startedAt < 850);
      for (const ripple of ripples) {
        const progress = Math.max(0, (timestamp - ripple.startedAt) / 850);
        brush.strokeStyle = `rgba(163, 21, 81, ${(1 - progress) * 0.35})`;
        brush.lineWidth = 1;
        brush.beginPath();
        brush.arc(ripple.x, ripple.y, 12 + (1 - Math.pow(1 - progress, 3)) * 190, 0, Math.PI * 2);
        brush.stroke();
      }

      if (!reducedMotion.matches && !document.hidden) {
        animationFrame = requestAnimationFrame(draw);
      }
    }

    function restart() {
      cancelAnimationFrame(animationFrame);
      lastFrameTime = 0;
      ripples = [];
      if (!document.hidden) animationFrame = requestAnimationFrame(draw);
    }

    function handlePointerMove(event: PointerEvent) {
      if (!finePointer.matches || reducedMotion.matches) return;
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      pointer.active = true;
    }

    function handlePointerDown(event: PointerEvent) {
      if (reducedMotion.matches || event.button !== 0) return;
      if (
        event.target instanceof Element &&
        event.target.closest('button, a, input, select, video, dialog')
      )
        return;
      ripples = [
        ...ripples.slice(-3),
        { x: event.clientX, y: event.clientY, startedAt: performance.now() },
      ];
      for (const particle of particles) {
        const deltaX = particle.x - event.clientX;
        const deltaY = particle.y - event.clientY;
        const distance = Math.hypot(deltaX, deltaY);
        if (distance > 0 && distance < 220) {
          const force = (1 - distance / 220) * 3;
          particle.impulseX += (deltaX / distance) * force;
          particle.impulseY += (deltaY / distance) * force;
        }
      }
    }

    function clearPointer() {
      pointer.active = false;
      pointer.x = -1000;
      pointer.y = -1000;
    }

    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('pointermove', handlePointerMove, { passive: true });
    window.addEventListener('pointerdown', handlePointerDown, { passive: true });
    document.documentElement.addEventListener('pointerleave', clearPointer);
    window.addEventListener('blur', clearPointer);
    document.addEventListener('visibilitychange', restart);
    reducedMotion.addEventListener('change', restart);
    finePointer.addEventListener('change', resize);

    return () => {
      cancelAnimationFrame(animationFrame);
      window.removeEventListener('resize', resize);
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerdown', handlePointerDown);
      document.documentElement.removeEventListener('pointerleave', clearPointer);
      window.removeEventListener('blur', clearPointer);
      document.removeEventListener('visibilitychange', restart);
      reducedMotion.removeEventListener('change', restart);
      finePointer.removeEventListener('change', resize);
    };
  }, []);

  return <canvas ref={canvasRef} className="particle-background" aria-hidden="true" />;
}
