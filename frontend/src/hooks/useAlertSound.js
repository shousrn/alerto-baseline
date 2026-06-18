import { useEffect, useRef } from 'react';

/**
 * useAlertSound — plays a Web Audio API beep when the driver state changes
 * to DROWSY or DISTRACTED. Uses a 2-second cooldown so it doesn't fire
 * every single polling cycle. No external audio file required.
 */
export function useAlertSound(state, enabled = true) {
  const audioCtxRef = useRef(null);
  const lastPlayedAt = useRef(0);
  const prevState = useRef(state);

  useEffect(() => {
    if (!enabled) return;

    const isAlert = state === 'DROWSY' || state === 'DISTRACTED';
    const stateChanged = state !== prevState.current;
    const now = Date.now();
    const cooldownMs = 2000;

    if (isAlert && (stateChanged || now - lastPlayedAt.current > cooldownMs)) {
      playBeep(state);
      lastPlayedAt.current = now;
    }

    prevState.current = state;
  }, [state, enabled]);

  function playBeep(state) {
    try {
      // Lazily create AudioContext on first interaction (browser policy)
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      const ctx = audioCtxRef.current;

      // Resume context if suspended (browser autoplay policy)
      if (ctx.state === 'suspended') {
        ctx.resume();
      }

      const isDrowsy = state === 'DROWSY';

      // Two-tone alert: a pair of short beeps
      const tones = isDrowsy
        ? [{ freq: 880, start: 0, duration: 0.18 }, { freq: 660, start: 0.22, duration: 0.18 }]
        : [{ freq: 1200, start: 0, duration: 0.12 }, { freq: 1200, start: 0.16, duration: 0.12 }, { freq: 1200, start: 0.32, duration: 0.20 }];

      const now = ctx.currentTime;

      tones.forEach(({ freq, start, duration }) => {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();

        osc.connect(gain);
        gain.connect(ctx.destination);

        osc.type = isDrowsy ? 'sine' : 'square';
        osc.frequency.setValueAtTime(freq, now + start);

        gain.gain.setValueAtTime(0, now + start);
        gain.gain.linearRampToValueAtTime(0.35, now + start + 0.01);
        gain.gain.exponentialRampToValueAtTime(0.001, now + start + duration);

        osc.start(now + start);
        osc.stop(now + start + duration + 0.05);
      });
    } catch (e) {
      // Silently fail if Web Audio API is unavailable
      console.warn('[ALERTO] Audio playback failed:', e.message);
    }
  }
}
