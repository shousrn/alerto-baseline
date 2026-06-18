import React from 'react';

const DriverView = ({ metrics }) => {
  const { state } = metrics;
  
  let config = {
    color: 'emerald',
    title: 'Alert',
    subtitle: 'Drive safely',
    bg: 'bg-emerald-500/5',
    border: 'border-emerald-500/20',
    pulse: 'bg-emerald-500',
    text: 'text-emerald-400',
    icon: (
      <svg viewBox="0 0 24 24" className="w-24 h-24 text-emerald-400" fill="none" stroke="currentColor" strokeWidth="1.5">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
        <path d="m9 12 2 2 4-4" />
      </svg>
    )
  };

  if (state === 'DROWSY') {
    config = {
      color: 'amber',
      title: 'Drowsy',
      subtitle: 'Pull over and rest',
      bg: 'bg-amber-500/5',
      border: 'border-amber-500/20',
      pulse: 'bg-amber-400',
      text: 'text-amber-400',
      icon: (
        <svg viewBox="0 0 24 24" className="w-24 h-24 text-amber-400" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M2 12h20" />
          <path d="M4 12v-2a8 8 0 0 1 16 0v2" />
          <path d="M9 4v2" />
          <path d="M15 4v2" />
          <path d="M12 2v2" />
          <text x="18" y="8" fontSize="6" fill="currentColor" stroke="none">z</text>
          <text x="22" y="4" fontSize="4" fill="currentColor" stroke="none">z</text>
        </svg>
      )
    };
  } else if (state === 'DISTRACTED') {
    config = {
      color: 'red',
      title: 'Distracted',
      subtitle: 'Eyes on the road',
      bg: 'bg-red-500/5',
      border: 'border-red-500/20',
      pulse: 'bg-red-500',
      text: 'text-red-500',
      icon: (
        <svg viewBox="0 0 24 24" className="w-24 h-24 text-red-500" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
          <path d="M12 9v4" />
          <path d="M12 17h.01" />
        </svg>
      )
    };
  }

  return (
    <div className={`absolute inset-0 flex flex-col items-center justify-center ${config.bg} transition-colors duration-500`}>
      <div className="absolute top-8 right-8">
        <div className={`inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium border ${config.border} ${config.text} bg-slate-950/50 backdrop-blur-md shadow-lg transition-colors duration-500`}>
          <span className="relative flex h-2.5 w-2.5">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${config.pulse}`}></span>
            <span className={`relative inline-flex rounded-full h-2.5 w-2.5 ${config.pulse}`}></span>
          </span>
          {config.title}
        </div>
      </div>

      <div className="relative">
        <div className={`absolute inset-0 rounded-full animate-ping opacity-20 ${config.pulse} blur-2xl`}></div>
        <div className={`relative p-12 rounded-full border ${config.border} bg-slate-950/80 backdrop-blur-xl shadow-2xl flex items-center justify-center transition-colors duration-500`}>
          {config.icon}
        </div>
      </div>

      <h1 className={`mt-12 text-7xl font-bold tracking-tight ${config.text} transition-colors duration-500 drop-shadow-md`}>
        {config.title}
      </h1>
      <p className="mt-6 text-2xl text-slate-400 font-medium tracking-wide">
        {config.subtitle}
      </p>
      
      {state !== 'ALERT' && (
        <div className="absolute bottom-16 text-slate-500 text-sm animate-pulse">
          Alert will clear automatically when normal driving resumes
        </div>
      )}
    </div>
  );
};

export default DriverView;
