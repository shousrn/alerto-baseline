import React from 'react';

const Landing = ({ onStart, onDriverView }) => {
  return (
    <div className="absolute inset-0 bg-slate-950 flex flex-col items-center justify-center overflow-hidden">
      {/* Background decoration */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-cyan-900/10 rounded-full blur-3xl pointer-events-none"></div>
      
      <div className="z-10 flex flex-col items-center max-w-2xl text-center px-6">
        <div className="mb-8 p-4 rounded-3xl bg-slate-900/50 border border-slate-800 shadow-2xl">
          <svg viewBox="0 0 32 32" className="w-16 h-16">
            <circle cx="16" cy="16" r="11" fill="none" stroke="#52525b" strokeWidth="1.5" />
            <circle cx="16" cy="16" r="3.5" fill="#22d3ee" className="animate-pulse" />
          </svg>
        </div>
        
        <h1 className="text-5xl font-bold tracking-tight text-white mb-4">
          ALERTO <span className="text-cyan-400">Baseline</span>
        </h1>
        
        <p className="text-lg text-slate-400 mb-10 leading-relaxed max-w-xl">
          Adaptive driver monitoring system with real-time biometric tracking
          and spatial classification. Fixed global thresholds — SVM (RBF kernel) on MediaPipe 478-pt landmarks.
        </p>
        
        <div className="flex items-center gap-4">
          <button 
            onClick={onStart}
            className="px-8 py-3.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold tracking-wide transition-colors shadow-[0_0_20px_rgba(34,211,238,0.3)] hover:shadow-[0_0_30px_rgba(34,211,238,0.5)] cursor-pointer"
          >
            Start Monitoring
          </button>
          
          <button 
            onClick={onDriverView}
            className="px-8 py-3.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-medium tracking-wide transition-colors border border-slate-700 cursor-pointer"
          >
            Driver HUD
          </button>
        </div>
      </div>
      
      <div className="absolute bottom-8 left-0 right-0 flex justify-center items-center gap-8 text-xs font-mono text-slate-500">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
          VISION CORE ONLINE
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
          NEURAL NET ACTIVE
        </div>
      </div>
    </div>
  );
};

export default Landing;
