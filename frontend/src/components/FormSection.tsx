import { useState, useEffect, type ReactNode } from "react";
import { Plus, Minus } from "lucide-react";

interface Props {
  id: string;
  title: string;
  index: number;
  fieldCount: number;
  filledCount: number;
  isOpen: boolean;
  onToggle: () => void;
  children: ReactNode;
}

export default function FormSection({
  id,
  title,
  index,
  fieldCount,
  filledCount,
  isOpen,
  onToggle,
  children,
}: Props) {
  const [mounted, setMounted] = useState(isOpen);

  useEffect(() => {
    if (isOpen) setMounted(true);
  }, [isOpen]);

  const done = filledCount > 0 && filledCount === fieldCount;
  const tagState = done ? "is-done" : isOpen ? "is-open" : "";

  return (
    <div className="border-b border-line last:border-b-0">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={isOpen}
        aria-controls={`section-${id}`}
        className="group flex w-full items-center gap-3 py-4 text-left"
      >
        <span className={`channel-tag ${tagState}`}>CH.{String(index).padStart(2, "0")}</span>
        <span className={`flex-1 font-display text-[16px] font-semibold tracking-wide ${isOpen ? "text-ink" : "text-muted"}`}>
          {title.toUpperCase()}
        </span>
        {!isOpen && filledCount > 0 && (
          <span className="font-mono text-[11px] tabular-nums text-muted">
            {filledCount}/{fieldCount}
          </span>
        )}
        {isOpen ? (
          <Minus className="h-3.5 w-3.5 text-copper" strokeWidth={2.5} />
        ) : (
          <Plus className="h-3.5 w-3.5 text-muted group-hover:text-copper" strokeWidth={2.5} />
        )}
      </button>

      {mounted && (
        <div
          id={`section-${id}`}
          className={`transition-all duration-150 ease-in-out ${
            isOpen ? "max-h-[5000px] opacity-100" : "max-h-0 overflow-hidden opacity-0"
          }`}
          onTransitionEnd={() => {
            if (!isOpen) setMounted(false);
          }}
        >
          <div className="border-l-2 border-line pb-6 pl-[22px]">
            <div className="grid grid-cols-1 gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
              {children}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}