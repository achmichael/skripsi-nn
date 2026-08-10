export default function Header() {
  return (
    <header className="border-b-2 border-circuit bg-ink">
      <div className="mx-auto flex max-w-6xl items-center gap-3 px-5 py-4 sm:px-8">
        <div className="flex h-8 w-8 items-center justify-center rounded-sm border border-brass/50">
          <div className="h-2 w-2 rounded-full bg-brass" />
        </div>
        <div>
          <p className="font-mono text-[10px] tracking-[0.2em] text-brass">PANEL·001</p>
          <p className="font-display text-[15px] font-semibold tracking-wide text-paper">
            PREDIKSI KONSUMSI LISTRIK
          </p>
        </div>
      </div>
    </header>
  );
}