import { Plus, Minus } from "lucide-react";
import type { ApplianceState } from "../types/ui";
import type { AppliancePreset } from "../data/appliancePresets";

interface ApplianceCardProps {
  title: string;
  icon: React.ReactNode;
  state: ApplianceState;
  presets?: AppliancePreset[];
  onChange: (state: ApplianceState) => void;
}

export default function ApplianceCard({ title, icon, state, presets, onChange }: ApplianceCardProps) {
  const handleJumlahChange = (delta: number) => {
    const newJumlah = Math.max(0, state.jumlah + delta);
    onChange({ ...state, jumlah: newJumlah });
  };

  const hasWattInput = presets !== undefined || state.isCustomWatt !== undefined;

  return (
    <div className={`p-5 rounded-3xl border transition-all duration-300
      ${state.jumlah > 0 
        ? 'bg-white border-teal-200 shadow-[0_8px_30px_rgb(0,0,0,0.04)] ring-1 ring-teal-50' 
        : 'bg-slate-50/50 border-slate-200 hover:border-slate-300'}`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className={`p-3 rounded-2xl ${state.jumlah > 0 ? 'bg-teal-50 text-teal-600' : 'bg-slate-100 text-slate-400'}`}>
            {icon}
          </div>
          <div>
            <h3 className={`font-semibold text-lg ${state.jumlah > 0 ? 'text-slate-800' : 'text-slate-500'}`}>{title}</h3>
            {state.jumlah > 0 && (
              <p className="text-xs font-medium text-teal-600 mt-0.5">
                {state.jumlah} Unit • {state.jam} Jam/Hari
                {hasWattInput && ` • ${state.watt} W`}
              </p>
            )}
          </div>
        </div>

        {/* Stepper */}
        <div className="flex items-center gap-3 bg-slate-100/80 rounded-full p-1 border border-slate-200/60">
          <button 
            type="button"
            onClick={() => handleJumlahChange(-1)}
            disabled={state.jumlah === 0}
            className="w-8 h-8 rounded-full flex items-center justify-center text-slate-600 bg-white shadow-sm disabled:opacity-50 disabled:shadow-none hover:bg-slate-50 transition-colors"
          >
            <Minus className="w-4 h-4" />
          </button>
          <span className="w-6 text-center font-bold text-slate-700">{state.jumlah}</span>
          <button 
            type="button"
            onClick={() => handleJumlahChange(1)}
            className="w-8 h-8 rounded-full flex items-center justify-center text-slate-600 bg-white shadow-sm hover:bg-slate-50 transition-colors"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Expanded Settings */}
      {state.jumlah > 0 && (
        <div className="mt-6 pt-5 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-top-4 duration-300">
          {/* Preset / Watt Selector */}
          {hasWattInput && (
            <div className="space-y-3">
              <label className="text-sm font-semibold text-slate-700 flex justify-between">
                <span>Daya (Watt) per unit</span>
                {presets && (
                  <button 
                    type="button"
                    onClick={() => onChange({ ...state, isCustomWatt: !state.isCustomWatt, watt: state.isCustomWatt ? presets[0].watt : state.watt })}
                    className="text-xs text-teal-600 hover:text-teal-700 underline underline-offset-2"
                  >
                    {state.isCustomWatt ? 'Gunakan Preset' : 'Isi Manual'}
                  </button>
                )}
              </label>
              
              {state.isCustomWatt ? (
                <div className="relative">
                  <input 
                    type="number"
                    min="0"
                    value={state.watt || ""}
                    onChange={(e) => onChange({ ...state, watt: Number(e.target.value) })}
                    className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-slate-700 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500 transition-all"
                    placeholder="Masukkan watt..."
                  />
                  <span className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 text-sm font-medium">W</span>
                </div>
              ) : presets && (
                <div className="grid grid-cols-2 gap-2">
                  {presets.map(p => (
                    <button
                      key={p.label}
                      type="button"
                      onClick={() => onChange({ ...state, watt: p.watt })}
                      className={`text-xs px-3 py-2.5 rounded-xl border font-medium transition-all text-left flex flex-col gap-0.5
                        ${state.watt === p.watt 
                          ? 'bg-teal-50 border-teal-500/30 text-teal-700 shadow-sm' 
                          : 'bg-white border-slate-200 text-slate-600 hover:border-teal-200 hover:bg-teal-50/50'}`}
                    >
                      <span className="truncate w-full">{p.label}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Jam Slider */}
          <div className="space-y-4">
            <label className="text-sm font-semibold text-slate-700 flex justify-between">
              <span>Durasi Pemakaian</span>
              <span className="text-teal-600 bg-teal-50 px-2 py-0.5 rounded-md font-bold">{state.jam} Jam/Hari</span>
            </label>
            <input 
              type="range"
              min="0"
              max="24"
              step="0.5"
              value={state.jam}
              onChange={(e) => onChange({ ...state, jam: Number(e.target.value) })}
              className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-teal-500"
            />
            <div className="flex justify-between text-[10px] font-medium text-slate-400 px-1">
              <span>0 Jam</span>
              <span>12 Jam</span>
              <span>24 Jam</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
