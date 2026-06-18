import React, { useState } from 'react';
import DeepAnalytics from './DeepAnalytics';

const MetricCard = ({ title, value, threshold, max, status, isWarning, icon }) => {
  const isDanger = status === 'High closure' || status === 'Yawning' || status === 'Off-axis';
  const progressPct = max ? Math.min(100, (value / max) * 100) : 0;
  
  return (
    <div className={`p-4 rounded-2xl bg-slate-900 border ${isWarning ? 'border-amber-500/50' : 'border-slate-800'} flex flex-col justify-between h-full relative overflow-hidden transition-colors`}>
      <div className="flex justify-between items-start mb-2">
        <div className="flex items-center gap-2">
          {icon}
          <span className="text-sm font-medium text-slate-400">{title}</span>
        </div>
        <div className={`px-2 py-0.5 rounded text-[11px] font-semibold tracking-wide uppercase ${isDanger ? 'bg-red-500/10 text-red-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
          {status}
        </div>
      </div>
      
      <div className="flex items-baseline gap-2 mt-auto">
        <span className="text-3xl font-bold text-white tabular-nums">{typeof value === 'number' ? value.toFixed(3) : value}</span>
        {threshold && (
          <span className="text-xs font-mono text-slate-500">THR {threshold.toFixed(3)}</span>
        )}
      </div>
      
      {max && (
        <div className="absolute bottom-0 left-0 right-0 h-1 bg-slate-800">
          <div 
            className={`h-full ${isDanger ? 'bg-red-500' : 'bg-cyan-500'}`} 
            style={{ width: `${progressPct}%`, transition: 'width 0.3s ease' }}
          ></div>
        </div>
      )}
    </div>
  );
};

const SystemDashboard = ({ metrics }) => {
  const [showAnalytics, setShowAnalytics] = useState(false);

  // EAR is close to threshold logic for amber highlight
  const isEarWarning = (metrics.ear - metrics.ear_threshold) < 0.05 && metrics.ear > metrics.ear_threshold;

  return (
    <div className="flex-1 flex flex-col min-h-0 bg-slate-950 p-4">
      <div className="flex justify-between items-center mb-4 px-2">
        <div className="text-sm font-medium text-slate-400 flex items-center gap-2">
          <span>PH Filipino baseline</span>
          <span className="w-1 h-1 rounded-full bg-slate-600"></span>
          <span>Subject P-001</span>
          <span className="w-1 h-1 rounded-full bg-slate-600"></span>
          <span className="font-mono text-xs">{metrics.fps} FPS</span>
          <span className="w-1 h-1 rounded-full bg-slate-600"></span>
          <span className="font-mono text-xs">{metrics.latency_ms}ms</span>
        </div>
        <button 
          onClick={() => setShowAnalytics(true)}
          className="px-4 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-sm font-medium text-slate-200 transition-colors border border-slate-700 cursor-pointer flex items-center gap-2"
        >
          <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 12V7H5a2 2 0 0 1 0-4h14v4" />
            <path d="M3 15h18v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4z" />
            <path d="M8 15v6" />
          </svg>
          Deep analytics
        </button>
      </div>

      <div className="flex-1 flex gap-4 min-h-0">
        {/* Left Panel - Video Feed */}
        <div className="w-[60%] flex flex-col relative rounded-3xl overflow-hidden bg-slate-900 border border-slate-800">
          <div className="absolute top-4 left-4 z-10 flex items-center gap-2 px-3 py-1.5 rounded-md bg-black/50 backdrop-blur text-xs font-medium text-white border border-white/10">
            <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></span>
            Live · Front camera
          </div>
          
          <img 
            src="/video_feed" 
            alt="Live feed" 
            className="w-full h-full object-cover" 
            key={Date.now()} // Force refresh if stream breaks, though MJPEG normally handles it
          />

          <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center px-4 py-2 rounded-lg bg-black/50 backdrop-blur text-xs font-mono text-slate-300 border border-white/10">
            <span>468 facial landmarks</span>
            <span>{metrics.fps} FPS · {metrics.latency_ms}ms latency</span>
          </div>
        </div>

        {/* Right Panel - Metric Cards */}
        <div className="w-[40%] flex flex-col gap-4">
          <MetricCard 
            title="Ocular Activity (EAR & PERCLOS)"
            value={metrics.ear}
            threshold={metrics.ear_threshold}
            max={0.5}
            status={metrics.perclos > metrics.perclos_threshold ? 'High closure' : 'Normal'}
            isWarning={isEarWarning}
            icon={
              <svg viewBox="0 0 24 24" className="w-5 h-5 text-cyan-500" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            }
          />
          
          <MetricCard 
            title="Oral Activity (MAR)"
            value={metrics.mar}
            threshold={metrics.mar_threshold}
            max={1.0}
            status={metrics.mar > metrics.mar_threshold ? 'Yawning' : 'Normal'}
            icon={
              <svg viewBox="0 0 24 24" className="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 20c4.4 0 8-3.6 8-8s-3.6-8-8-8-8 3.6-8 8 3.6 8 8 8z" />
                <path d="M8 12h8" />
              </svg>
            }
          />
          
          <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col justify-between h-full">
            <div className="flex justify-between items-start mb-2">
              <div className="flex items-center gap-2">
                <svg viewBox="0 0 24 24" className="w-5 h-5 text-emerald-500" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
                </svg>
                <span className="text-sm font-medium text-slate-400">Spatial Tracking</span>
              </div>
          <div className={`px-2 py-0.5 rounded text-[11px] font-semibold tracking-wide uppercase ${Math.abs(metrics.pitch) > (metrics.pitch_threshold || 20) || Math.abs(metrics.yaw) > (metrics.yaw_threshold || 30) ? 'bg-red-500/10 text-red-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
                {Math.abs(metrics.pitch) > (metrics.pitch_threshold || 20) || Math.abs(metrics.yaw) > (metrics.yaw_threshold || 30) ? 'Off-axis' : 'Contained'}
              </div>
            </div>
            
            <div className="grid grid-cols-3 gap-2 mt-auto">
              <div className="flex flex-col">
                <span className="text-xs text-slate-500 mb-1">PITCH</span>
                <span className="text-xl font-bold text-white tabular-nums">{metrics.pitch.toFixed(1)}°</span>
              </div>
              <div className="flex flex-col">
                <span className="text-xs text-slate-500 mb-1">YAW</span>
                <span className="text-xl font-bold text-white tabular-nums">{metrics.yaw.toFixed(1)}°</span>
              </div>
              <div className="flex flex-col">
                <span className="text-xs text-slate-500 mb-1">ROLL</span>
                <span className="text-xl font-bold text-white tabular-nums">{metrics.roll.toFixed(1)}°</span>
              </div>
            </div>
          </div>
          
          <div className="p-4 rounded-2xl bg-slate-900 border border-slate-800 flex flex-col justify-between h-[120px] shrink-0">
            <div className="flex justify-between items-start">
              <div className="flex items-center gap-2">
                <svg viewBox="0 0 24 24" className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
                </svg>
                <span className="text-sm font-medium text-slate-400">Edge Performance</span>
              </div>
              <div className="px-2 py-0.5 rounded text-[11px] font-semibold tracking-wide uppercase bg-emerald-500/10 text-emerald-400">
                Healthy
              </div>
            </div>
            <div className="flex items-baseline gap-4 mt-auto">
              <div><span className="text-2xl font-bold text-white tabular-nums">{metrics.fps}</span><span className="text-xs text-slate-500 ml-1">FPS</span></div>
              <div><span className="text-2xl font-bold text-white tabular-nums">{metrics.latency_ms}</span><span className="text-xs text-slate-500 ml-1">ms</span></div>
            </div>
          </div>
        </div>
      </div>

      {showAnalytics && <DeepAnalytics metrics={metrics} onClose={() => setShowAnalytics(false)} />}
    </div>
  );
};

export default SystemDashboard;
