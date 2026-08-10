import type { ModelType, PredictionResponse } from "../types/prediction";
import GaugeDial from "./GaugeDial";

interface Props {
  modelType: ModelType;
  result: PredictionResponse | null;
  error: string | null;
  onRetry: () => void;
  filledCount: number;
  totalCount: number;
}

function formatRupiah(value: number): string {
  return new Intl.NumberFormat("id-ID", {
    style: "currency",
    currency: "IDR",
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

export default function ResultCard({ modelType, result, error, onRetry, filledCount, totalCount }: Props) {
  const pct = totalCount > 0 ? (filledCount / totalCount) * 100 : 0;

  return (
    <div className="space-y-5 rounded-lg border border-line bg-panel p-5">
      <div>
        <p className="channel-tag mb-3">PROGRESS</p>
        <GaugeDial percent={pct} />
        <p className="mt-1 text-center font-mono text-[11px] text-muted">
          {filledCount} / {totalCount} FIELD TERISI
        </p>
      </div>

      <div className="h-px bg-line" />

      <div>
        <p className="channel-tag mb-3">OUTPUT</p>

        {error ? (
          <div className="lcd-screen p-4">
            <p className="font-mono text-[13px] font-medium text-red-400">ERR // GAGAL DIPROSES</p>
            <p className="mt-1 font-mono text-[11px] leading-relaxed text-red-300/70">
              Periksa kembali input, lalu coba lagi.
            </p>
            <button
              type="button"
              onClick={onRetry}
              className="btn-press mt-3 font-mono text-[11px] font-medium text-red-400 underline underline-offset-2"
            >
              ULANGI →
            </button>
          </div>
        ) : !result ? (
          <div id="result-card" className="lcd-screen flex min-h-[92px] items-center justify-center p-4">
            <p className="text-center font-mono text-[11px] leading-relaxed text-lcd-fg/40">
              MENUNGGU DATA · JALANKAN PREDIKSI
            </p>
          </div>
        ) : (
          <div id="result-card" className="lcd-screen p-4">
            <p className="font-mono text-[10px] tracking-widest text-lcd-fg/60">
              {modelType === "prepaid" ? "ESTIMASI DURASI TOKEN" : "ESTIMASI TAGIHAN"}
            </p>
            <p className="mt-1 font-display text-[36px] font-bold leading-none tracking-tight text-lcd-fg">
              {modelType === "prepaid" ? `${result.prediction} HR` : formatRupiah(result.prediction)}
            </p>
            <p className="mt-2 font-mono text-[10px] leading-relaxed text-lcd-fg/50">
              {modelType === "prepaid"
                ? "berdasarkan pola konsumsi yang diinput"
                : "berdasarkan data penggunaan bulan ini"}
            </p>
          </div>
        )}
      </div>

      <div className="h-px bg-line" />

      <div>
        <p className="channel-tag mb-2">MODEL</p>
        <p className="font-display text-[15px] font-semibold text-ink">
          {modelType === "prepaid" ? "PRABAYAR" : "PASCABAYAR"}
        </p>
        <p className="mt-0.5 text-[12px] text-muted">
          {modelType === "prepaid" ? "Prediksi durasi penggunaan token listrik." : "Estimasi biaya listrik bulanan."}
        </p>
      </div>
    </div>
  );
}