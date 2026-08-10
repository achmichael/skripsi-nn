import { ArrowRight } from "lucide-react";
import LoadingSpinner from "./LoadingSpinner";

interface Props {
  isLoading: boolean;
  onPredict: () => void;
  onReset: () => void;
}

export default function PredictionActions({ isLoading, onPredict, onReset }: Props) {
  return (
    <div className="flex items-center gap-4 pt-2">
      <button
        type="button"
        onClick={onPredict}
        disabled={isLoading}
        className="inline-flex items-center gap-2 bg-slate-900 px-6 py-2.5 text-[13px] font-medium text-white transition-all hover:bg-slate-800 active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-40"
      >
        {isLoading ? (
          <>
            <LoadingSpinner />
            <span>Memproses...</span>
          </>
        ) : (
          <>
            <span>Jalankan Prediksi</span>
            <ArrowRight className="h-3.5 w-3.5" strokeWidth={2} />
          </>
        )}
      </button>
      <button
        type="button"
        onClick={onReset}
        disabled={isLoading}
        className="text-[13px] text-slate-400 transition-colors hover:text-slate-600 disabled:opacity-40"
      >
        Reset
      </button>
    </div>
  );
}
