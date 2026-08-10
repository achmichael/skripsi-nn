import type { ModelType } from "../types/prediction";

interface Props {
  modelType: ModelType;
  onChange: (type: ModelType) => void;
}

export default function ModelSelector({ modelType, onChange }: Props) {
  const options: { value: ModelType; label: string }[] = [
    { value: "prepaid", label: "PRABAYAR" },
    { value: "postpaid", label: "PASCABAYAR" },
  ];

  return (
    <div className="inline-flex rounded-md border border-line bg-panel p-1">
      {options.map((opt) => {
        const active = modelType === opt.value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => onChange(opt.value)}
            className={`btn-press rounded-[5px] px-4 py-2 font-mono text-[12px] font-semibold tracking-wide transition-colors ${
              active ? "bg-circuit text-paper" : "text-muted hover:text-ink"
            }`}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}