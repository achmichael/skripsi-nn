interface Props {
  percent: number; // 0–100
  size?: number;
}

export default function GaugeDial({ percent, size = 176 }: Props) {
  const clamped = Math.max(0, Math.min(100, percent));
  const angle = -90 + (clamped / 100) * 180; // -90 (left) → 90 (right)
  const cx = 100;
  const cy = 108;
  const rNeedle = 66;
  const rTick = 80;

  const ticks = Array.from({ length: 11 }, (_, i) => {
    const a = -90 + (i / 10) * 180;
    const rad = (a * Math.PI) / 180;
    const x1 = cx + Math.cos(rad) * (rTick - 8);
    const y1 = cy + Math.sin(rad) * (rTick - 8);
    const x2 = cx + Math.cos(rad) * rTick;
    const y2 = cy + Math.sin(rad) * rTick;
    return { x1, y1, x2, y2, major: i % 5 === 0 };
  });

  return (
    <div className="flex flex-col items-center">
      <svg width={size} height={size * 0.62} viewBox="0 0 200 120" aria-hidden="true">
        <path
          d="M 20 108 A 80 80 0 0 1 180 108"
          fill="none"
          stroke="var(--color-line)"
          strokeWidth="10"
          strokeLinecap="round"
        />
        <path
          d="M 20 108 A 80 80 0 0 1 180 108"
          fill="none"
          stroke="var(--color-copper)"
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${(clamped / 100) * 251} 251`}
          className="transition-all duration-700 ease-out"
        />
        {ticks.map((t, i) => (
          <line
            key={i}
            x1={t.x1}
            y1={t.y1}
            x2={t.x2}
            y2={t.y2}
            stroke="var(--color-muted)"
            strokeWidth={t.major ? 1.5 : 1}
            opacity={t.major ? 0.6 : 0.3}
          />
        ))}
        <g
          className="origin-center transition-transform duration-700 ease-out"
          style={{ transformOrigin: `${cx}px ${cy}px`, transform: `rotate(${angle}deg)` }}
        >
          <line x1={cx} y1={cy} x2={cx - rNeedle} y2={cy} stroke="var(--color-ink)" strokeWidth="3" strokeLinecap="round" />
        </g>
        <circle cx={cx} cy={cy} r="6" fill="var(--color-ink)" />
        <circle cx={cx} cy={cy} r="2.5" fill="var(--color-brass)" />
      </svg>
      <div className="-mt-2 flex items-baseline gap-1 font-display text-[34px] font-bold leading-none text-ink">
        {Math.round(clamped)}
        <span className="text-[14px] font-mono font-normal text-muted">%</span>
      </div>
    </div>
  );
}