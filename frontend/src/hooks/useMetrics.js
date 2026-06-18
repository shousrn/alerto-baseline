import { useState, useEffect, useRef } from 'react';

/** Default values mirror the /metrics JSON schema (Module 4.4). */
const DEFAULT_METRICS = {
  state: 'ALERT',
  ear: 0.0,
  perclos: 0.0,
  mar: 0.0,
  pitch: 0.0,
  yaw: 0.0,
  roll: 0.0,
  fps: 0,
  latency_ms: 0,
  ear_threshold: 0.25,
  mar_threshold: 0.60,
  perclos_threshold: 30.0,
  pitch_threshold: 20.0,
  yaw_threshold: 30.0,
  svm_proba: { ALERT: 0.0, DROWSY: 0.0, DISTRACTED: 0.0 },
  ear_consec: 0,
  mar_consec: 0,
  pose_consec: 0,
  error_message: null,
};

/**
 * useMetrics — Module 5.1
 *
 * Polls /metrics every 300 ms. Returns:
 *   metrics   : latest JSON payload (or DEFAULT_METRICS before first fetch)
 *   error     : last fetch error, or null
 *   isStale   : true when no successful fetch has occurred in the last 2 s
 */
export function useMetrics() {
  const [metrics, setMetrics] = useState(DEFAULT_METRICS);
  const [error, setError] = useState(null);
  const [isStale, setIsStale] = useState(false);
  const lastSuccessAt = useRef(Date.now());

  useEffect(() => {
    let active = true;

    const fetchMetrics = async () => {
      try {
        const response = await fetch('/metrics');
        if (!response.ok) {
          throw new Error(`HTTP error: ${response.status}`);
        }
        const data = await response.json();
        if (active) {
          setMetrics(data);
          setError(null);
          lastSuccessAt.current = Date.now();
          setIsStale(false);
        }
      } catch (e) {
        if (active) {
          setError(e);
          // Mark stale if last success was > 2 seconds ago
          if (Date.now() - lastSuccessAt.current > 2000) {
            setIsStale(true);
          }
        }
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 300);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  return { metrics, error, isStale };
}
