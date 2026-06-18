import { useState, useEffect } from 'react';
import Landing from './components/Landing';
import SystemDashboard from './components/SystemDashboard';
import DriverView from './components/DriverView';
import { useMetrics } from './hooks/useMetrics';
import { useAlertSound } from './hooks/useAlertSound';

function App() {
  const [tab, setTab] = useState('LANDING');
  const { metrics } = useMetrics();

  // ── Theme ───────────────────────────────────────────────────────────────
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('alerto-theme') || 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('alerto-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(t => (t === 'dark' ? 'light' : 'dark'));

  // ── Audio alert ─────────────────────────────────────────────────────────
  const [soundEnabled, setSoundEnabled] = useState(() => {
    return localStorage.getItem('alerto-sound') !== 'off';
  });

  useAlertSound(metrics.state, soundEnabled);

  const toggleSound = () => {
    setSoundEnabled(prev => {
      localStorage.setItem('alerto-sound', prev ? 'off' : 'on');
      return !prev;
    });
  };

  // ── Styles ──────────────────────────────────────────────────────────────
  const navClass = 'flex items-center gap-1 relative';
  const btnClass = (active) =>
    `px-4 py-2 rounded-lg text-[13px] font-medium transition-colors border-none cursor-pointer ${
      active
        ? 'text-slate-100 bg-white/10'
        : 'text-slate-500 hover:text-slate-300 hover:bg-white/5'
    }`;

  const renderTopChrome = () => (
    <header className="shrink-0 px-8 flex items-center justify-between z-30 border-b border-slate-800/70 h-[64px] bg-slate-950/80 backdrop-blur-md">
      <div className="flex items-center gap-10">
        <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => setTab('LANDING')}>
          <svg viewBox="0 0 32 32" className="w-6 h-6">
            <circle cx="16" cy="16" r="11" fill="none" stroke="#52525b" strokeWidth="1.5" />
            <circle cx="16" cy="16" r="3.5" fill="#22d3ee" />
          </svg>
          <span className="text-[15px] font-medium tracking-tight text-slate-50">ALERTO</span>
          <span className="ml-2 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-semibold tracking-wide uppercase bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            Baseline
          </span>
        </div>

        <nav className={navClass}>
          <button className={btnClass(tab === 'LANDING')} onClick={() => setTab('LANDING')}>Home</button>
          <button className={btnClass(tab === 'SYSTEM')} onClick={() => setTab('SYSTEM')}>System</button>
          <button className={btnClass(tab === 'DRIVER')} onClick={() => setTab('DRIVER')}>Driver</button>
        </nav>
      </div>

      <div className="flex items-center gap-3">
        <div className="text-right">
          <div className="text-[12px] text-slate-300">SVM (RBF · Global Scaler)</div>
          <div className="text-[10.5px] text-slate-500 font-mono tabular">COSC 304 · Group 7</div>
        </div>

        {/* Sound toggle */}
        <button
          onClick={toggleSound}
          title={soundEnabled ? 'Mute alerts' : 'Unmute alerts'}
          aria-label="Toggle sound"
          className="w-9 h-9 rounded-full hover:bg-slate-800 flex items-center justify-center transition border border-slate-800 cursor-pointer"
          style={{ color: soundEnabled ? '#22d3ee' : '#64748b' }}
        >
          {soundEnabled ? (
            <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
              <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
              <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2">
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
              <line x1="23" y1="9" x2="17" y2="15" />
              <line x1="17" y1="9" x2="23" y2="15" />
            </svg>
          )}
        </button>

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          aria-label="Toggle theme"
          className="w-9 h-9 rounded-full hover:bg-slate-800 text-slate-400 hover:text-slate-200 flex items-center justify-center transition border border-slate-800 cursor-pointer"
        >
          {theme === 'light' ? (
            <svg viewBox="0 0 24 24" className="w-4 h-4" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 12.5A9 9 0 1 1 11.5 3 7 7 0 0 0 21 12.5z" fill="currentColor" stroke="none"/>
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" className="w-4 h-4">
              <circle cx="12" cy="12" r="4" fill="currentColor" />
              {[0, 1, 2, 3, 4, 5, 6, 7].map(i => {
                const a = (i / 8) * Math.PI * 2;
                return (
                  <line
                    key={i}
                    x1={12 + Math.cos(a) * 7}  y1={12 + Math.sin(a) * 7}
                    x2={12 + Math.cos(a) * 9.5} y2={12 + Math.sin(a) * 9.5}
                    stroke="currentColor" strokeWidth="1.7" strokeLinecap="round"
                  />
                );
              })}
            </svg>
          )}
        </button>

        {/* State badge */}
        <div className={`inline-flex items-center gap-2 rounded-full px-3.5 py-1.5 text-[13px] font-medium transition cursor-pointer select-none
          ${metrics.state === 'ALERT'       ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' :
            metrics.state === 'DROWSY'      ? 'bg-amber-500/10  text-amber-400  border border-amber-500/20'  :
                                              'bg-red-500/10    text-red-400    border border-red-500/20'}`}
        >
          <span className="relative inline-flex">
            <span className={`absolute inset-0 rounded-full animate-ping opacity-75 ${
              metrics.state === 'ALERT' ? 'bg-emerald-400' : metrics.state === 'DROWSY' ? 'bg-amber-400' : 'bg-red-500'
            }`} />
            <span className={`relative w-2 h-2 rounded-full ${
              metrics.state === 'ALERT' ? 'bg-emerald-400' : metrics.state === 'DROWSY' ? 'bg-amber-400' : 'bg-red-500'
            }`} />
          </span>
          {metrics.state === 'ALERT' ? 'Alert' : metrics.state === 'DROWSY' ? 'Drowsy' : 'Distracted'}
        </div>
      </div>
    </header>
  );

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-950 text-slate-200 font-sans">
      {tab !== 'LANDING' && renderTopChrome()}

      <main className="flex-1 flex flex-col relative min-h-0">
        {tab === 'LANDING' && <Landing onStart={() => setTab('SYSTEM')} onDriverView={() => setTab('DRIVER')} />}
        {tab === 'SYSTEM'  && <SystemDashboard metrics={metrics} />}
        {tab === 'DRIVER'  && <DriverView metrics={metrics} />}
      </main>
    </div>
  );
}

export default App;
