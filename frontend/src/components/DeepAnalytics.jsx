import React, { useState, useEffect, useRef } from 'react';

// Simple sparkline component
const Sparkline = ({ data, color, max }) => {
  const points = data.map((d, i) => {
    const x = (i / (data.length - 1)) * 100;
    const y = 100 - (Math.min(d, max) / max) * 100;
    return `${x},${y}`;
  }).join(' ');

  return (
    <div className="h-12 w-full relative pt-2">
      <svg className="w-full h-full overflow-visible" preserveAspectRatio="none">
        <polyline
          points={points}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
};

const DeepAnalytics = ({ metrics, onClose }) => {
  const [history, setHistory] = useState({ fps: [], latency: [] });
  const [events, setEvents] = useState([]);
  const lastStateRef = useRef(metrics.state);

  // Update history arrays
  useEffect(() => {
    setHistory(prev => {
      const newFps = [...prev.fps, metrics.fps].slice(-60);
      const newLatency = [...prev.latency, metrics.latency_ms].slice(-60);
      return { fps: newFps, latency: newLatency };
    });
    
    // Check for state transitions to add to event log
    if (metrics.state !== lastStateRef.current) {
      const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
      let message = '';
      if (metrics.state === 'DROWSY') message = 'Drowsiness detected (Threshold breached)';
      else if (metrics.state === 'DISTRACTED') message = 'Distraction detected (Off-axis head pose)';
      else message = 'Driver state recovered to ALERT';
      
      setEvents(prev => [{
        time: timestamp,
        state: metrics.state,
        message
      }, ...prev].slice(0, 20)); // Keep last 20 per Module 5.5 spec
      
      lastStateRef.current = metrics.state;
    }
  }, [metrics.fps, metrics.latency_ms, metrics.state]);

  const p_alert = metrics.svm_proba?.ALERT || 0;
  const p_drowsy = metrics.svm_proba?.DROWSY || 0;
  const p_distracted = metrics.svm_proba?.DISTRACTED || 0;

  return (
    <div className="absolute inset-0 z-50 flex justify-end bg-black/40 backdrop-blur-sm transition-opacity">
      <div className="w-[500px] h-full bg-slate-900 border-l border-slate-800 flex flex-col shadow-2xl animate-in slide-in-from-right-full duration-300">
        
        {/* Header */}
        <div className="p-6 border-b border-slate-800 shrink-0 flex justify-between items-start">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-xl font-bold text-white tracking-tight">Deep analytics</h2>
              <div className={`px-2 py-0.5 rounded text-[11px] font-semibold tracking-wide uppercase ${metrics.state === 'ALERT' ? 'bg-emerald-500/10 text-emerald-400' : metrics.state === 'DROWSY' ? 'bg-amber-500/10 text-amber-400' : 'bg-red-500/10 text-red-400'}`}>
                {metrics.state}
              </div>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed max-w-sm">
              Live SVM evaluation, biometric feature streams, edge performance and confusion matrix for the PH Filipino baseline
            </p>
          </div>
          <button onClick={onClose} className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors cursor-pointer">
            <svg viewBox="0 0 24 24" className="w-5 h-5" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
          
          {/* SVM Probabilities */}
          <div className="space-y-3">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">SVM Classification Proba</h3>
            
            <div className="flex flex-col gap-2">
              <div className="flex items-center gap-3">
                <span className="w-20 text-xs font-medium text-slate-300">ALERT</span>
                <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-emerald-500 transition-all duration-300" style={{ width: `${p_alert * 100}%` }}></div>
                </div>
                <span className="w-10 text-right text-xs font-mono text-slate-400">{(p_alert * 100).toFixed(0)}%</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-20 text-xs font-medium text-slate-300">DROWSY</span>
                <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-amber-500 transition-all duration-300" style={{ width: `${p_drowsy * 100}%` }}></div>
                </div>
                <span className="w-10 text-right text-xs font-mono text-slate-400">{(p_drowsy * 100).toFixed(0)}%</span>
              </div>
              <div className="flex items-center gap-3">
                <span className="w-20 text-xs font-medium text-slate-300">DISTRACTED</span>
                <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-red-500 transition-all duration-300" style={{ width: `${p_distracted * 100}%` }}></div>
                </div>
                <span className="w-10 text-right text-xs font-mono text-slate-400">{(p_distracted * 100).toFixed(0)}%</span>
              </div>
            </div>
          </div>

          <hr className="border-slate-800" />

          {/* Sub Panels */}
          <div className="grid grid-cols-2 gap-4">
            <div className="p-3 rounded-xl bg-slate-950/50 border border-slate-800">
              <span className="block text-[10px] font-bold text-slate-500 mb-2 uppercase tracking-wide">Ocular</span>
              <div className="mb-2">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">PERCLOS</span>
                  <span className="font-mono text-white">{metrics.perclos.toFixed(1)}%</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-cyan-500" style={{ width: `${Math.min(100, metrics.perclos)}%` }}></div>
                </div>
              </div>
              <div className="flex justify-between items-baseline mt-3">
                <span className="text-[10px] text-slate-500">EAR_LIVE</span>
                <span className="text-sm font-mono text-white">{metrics.ear.toFixed(3)}</span>
              </div>
              <div className="flex justify-between items-baseline">
                <span className="text-[10px] text-slate-500">BASELINE</span>
                <span className="text-xs font-mono text-slate-400">{metrics.ear_threshold.toFixed(3)}</span>
              </div>
            </div>

            <div className="p-3 rounded-xl bg-slate-950/50 border border-slate-800">
              <span className="block text-[10px] font-bold text-slate-500 mb-2 uppercase tracking-wide">Oral</span>
              <div className="mb-2">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">MAR</span>
                  <span className="font-mono text-white">{metrics.mar.toFixed(3)}</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-purple-500" style={{ width: `${Math.min(100, (metrics.mar/metrics.mar_threshold)*100)}%` }}></div>
                </div>
              </div>
              <div className="flex justify-between items-baseline mt-3">
                <span className="text-[10px] text-slate-500">YAWNS</span>
                <span className="text-sm font-mono text-white">{metrics.mar_consec}</span>
              </div>
              <div className="flex justify-between items-baseline">
                <span className="text-[10px] text-slate-500">BASELINE</span>
                <span className="text-xs font-mono text-slate-400">{metrics.mar_threshold.toFixed(3)}</span>
              </div>
            </div>
          </div>

          <div className="p-3 rounded-xl bg-slate-950/50 border border-slate-800">
            <div className="flex justify-between items-start mb-2">
              <span className="block text-[10px] font-bold text-slate-500 uppercase tracking-wide">Spatial Tracking</span>
              {(Math.abs(metrics.pitch) > (metrics.pitch_threshold || 20) || Math.abs(metrics.yaw) > (metrics.yaw_threshold || 30)) && (
                <span className="px-1.5 py-[1px] bg-red-500/20 text-red-400 text-[9px] font-bold rounded">BREACH</span>
              )}
            </div>
            <div className="flex items-center justify-between">
              {/* Simple radar crosshair */}
              <div className="w-16 h-16 rounded-full border border-slate-700 relative flex items-center justify-center shrink-0">
                <div className="absolute w-full h-[1px] bg-slate-800"></div>
                <div className="absolute h-full w-[1px] bg-slate-800"></div>
                <div className="absolute w-2 h-2 rounded-full bg-emerald-400 transition-all" style={{
                  transform: `translate(${(metrics.yaw/45)*24}px, ${(metrics.pitch/45)*24}px)`
                }}></div>
              </div>
              
              <div className="flex gap-4 text-right">
                <div>
                  <div className="text-[10px] text-slate-500">PITCH</div>
                  <div className="font-mono text-sm text-white">{metrics.pitch.toFixed(1)}°</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500">YAW</div>
                  <div className="font-mono text-sm text-white">{metrics.yaw.toFixed(1)}°</div>
                </div>
                <div>
                  <div className="text-[10px] text-slate-500">ROLL</div>
                  <div className="font-mono text-sm text-white">{metrics.roll.toFixed(1)}°</div>
                </div>
              </div>
            </div>
          </div>

          <hr className="border-slate-800" />

          {/* Performance Sparklines */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="flex justify-between items-baseline mb-1">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wide">FPS Rate</span>
                <span className="font-mono text-xs text-white">{metrics.fps}</span>
              </div>
              <Sparkline data={history.fps.length ? history.fps : [30,30]} color="#10b981" max={60} />
            </div>
            <div>
              <div className="flex justify-between items-baseline mb-1">
                <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wide">Latency</span>
                <span className="font-mono text-xs text-white">{metrics.latency_ms}ms</span>
              </div>
              <Sparkline data={history.latency.length ? history.latency : [30,30]} color="#3b82f6" max={100} />
            </div>
          </div>

          <hr className="border-slate-800" />

          {/* Event Log */}
          <div className="flex-1 flex flex-col min-h-0">
            <span className="block text-[10px] font-bold text-slate-500 mb-3 uppercase tracking-wide">Event Log</span>
            <div className="flex-1 overflow-y-auto space-y-2 pr-2">
              {events.length === 0 ? (
                <div className="text-xs text-slate-600 italic">No events recorded yet...</div>
              ) : (
                events.map((ev, i) => (
                  <div key={i} className="flex gap-3 text-xs">
                    <span className="font-mono text-slate-500 shrink-0">{ev.time}</span>
                    <div className="flex flex-col">
                      <span className={`${ev.state === 'ALERT' ? 'text-emerald-400' : ev.state === 'DROWSY' ? 'text-amber-400' : 'text-red-400'} font-medium`}>
                        {ev.state}
                      </span>
                      <span className="text-slate-400">{ev.message}</span>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

        </div>
      </div>
    </div>
  );
};

export default DeepAnalytics;
