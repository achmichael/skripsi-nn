import type { FieldConfig } from "../types/prediction";

interface Props {
  field: FieldConfig;
  value: string | number | boolean;
  error?: string;
  onChange: (name: string, value: string | number | boolean) => void;
}

export default function FormField({ field, value, error, onChange }: Props) {
  const inputId = field.name;

  const inputCls = [
    "block w-full border-0 border-b bg-transparent pb-2 pt-1 text-[14px] text-slate-900",
    "placeholder:text-slate-300 transition-colors",
    "focus:border-blue-600 focus:outline-none",
    error ? "border-red-400" : "border-slate-200",
  ].join(" ");

  return (
    <div>
      <label htmlFor={inputId} className="mb-1 block text-[12px] font-medium uppercase tracking-wide text-slate-400">
        {field.label}
        {field.required && <span className="ml-0.5 text-red-400">*</span>}
      </label>

      {field.type === "toggle" ? (
        <div className="flex items-center gap-3 pt-1">
          <button
            id={inputId}
            type="button"
            role="switch"
            aria-checked={Boolean(value)}
            onClick={() => onChange(field.name, !value)}
            className={`relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-500 ${
              value ? "bg-blue-600" : "bg-slate-200"
            }`}
          >
            <span
              className={`pointer-events-none inline-block h-4 w-4 translate-y-[2px] rounded-full bg-white shadow-sm transition-transform ${
                value ? "translate-x-[18px]" : "translate-x-[2px]"
              }`}
            />
          </button>
          <span className="text-[13px] text-slate-600">{value ? "Ya" : "Tidak"}</span>
        </div>
      ) : field.type === "select" ? (
        <select
          id={inputId}
          name={field.name}
          value={String(value)}
          onChange={(e) => onChange(field.name, e.target.value)}
          className={inputCls}
        >
          {field.options?.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      ) : field.type === "text" ? (
        <input
          id={inputId}
          name={field.name}
          type="text"
          value={String(value)}
          placeholder={field.placeholder}
          onChange={(e) => onChange(field.name, e.target.value)}
          className={inputCls}
        />
      ) : (
        <input
          id={inputId}
          name={field.name}
          type="number"
          value={value === "" ? "" : Number(value)}
          placeholder={field.placeholder}
          min={field.min}
          max={field.max}
          step={field.step}
          onChange={(e) => onChange(field.name, e.target.value === "" ? "" : Number(e.target.value))}
          className={inputCls}
        />
      )}

      {field.helperText && !error && (
        <p className="mt-1.5 text-[11px] leading-normal text-slate-400">{field.helperText}</p>
      )}
      {error && <p className="mt-1.5 text-[11px] text-red-500">{error}</p>}
    </div>
  );
}
