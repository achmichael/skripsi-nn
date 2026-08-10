import { Activity } from "lucide-react";
import type { UIState } from "../types/ui";
import type { ModelType } from "../types/prediction";

interface EnergySummarySidebarProps {
  uiState: UIState;
  modelType: ModelType;
}

export default function EnergySummarySidebar({ uiState }: EnergySummarySidebarProps) {
  // Hitung total energi
  let totalMain = 0;
  
  // Fungsi helper untuk menghitung kWh
  const calcKwh = (jumlah: number, watt: number, jam: number) => {
    return (jumlah * watt * jam) / 1000;
  };

  totalMain += calcKwh(uiState.kulkas.jumlah, uiState.kulkas.watt, uiState.kulkas.jam);
  totalMain += calcKwh(uiState.tv.jumlah, uiState.tv.watt, uiState.tv.jam);
  totalMain += calcKwh(uiState.ac.jumlah, uiState.ac.watt, uiState.ac.jam);
  totalMain += calcKwh(uiState.kipas.jumlah, uiState.kipas.watt, uiState.kipas.jam);
  totalMain += calcKwh(uiState.ricecooker.jumlah, uiState.ricecooker.watt, uiState.ricecooker.jam);

  // Mesin cuci khusus
  const mc = uiState.mesincuci;
  if (mc.jumlah > 0 && mc.frekuensiPerMinggu > 0) {
    const watt = mc.watt > 0 ? mc.watt : 0;
    const durasi = mc.jam > 0 ? mc.jam : 1;
    totalMain += (mc.jumlah * watt * mc.frekuensiPerMinggu * durasi) / (7 * 1000);
  }

  let totalAlatLain = 0;
  if (uiState.alatLainAda) {
    totalAlatLain = (uiState.alatLainTotalWatt * uiState.alatLainTotalJam) / 1000;
  }

  const totalAllKwhHari = totalMain + totalAlatLain;
  const totalAllKwhBulan = totalAllKwhHari * 30;

  return (
    <div className="bg-white rounded-3xl border border-teal-100 shadow-[0_8px_30px_rgb(0,0,0,0.04)] overflow-hidden">
      <div className="bg-teal-600/5 border-b border-teal-100 p-5">
        <h3 className="font-bold text-teal-800 flex items-center gap-2">
          <Activity className="w-5 h-5 text-teal-600" />
          Estimasi Real-time
        </h3>
        <p className="text-xs text-slate-500 mt-1">Dihitung otomatis berdasarkan input peralatan</p>
      </div>
      
      <div className="p-5 space-y-6">
        <div>
          <p className="text-sm font-semibold text-slate-500 mb-1">Total Konsumsi Harian</p>
          <div className="flex items-end gap-2">
            <span className="text-3xl font-black text-slate-800 tracking-tight">
              {totalAllKwhHari.toFixed(2)}
            </span>
            <span className="text-sm font-bold text-slate-400 mb-1">kWh/hari</span>
          </div>
        </div>

        <div className="h-px bg-slate-100 w-full" />

        <div>
          <p className="text-sm font-semibold text-slate-500 mb-1">Estimasi Bulanan</p>
          <div className="flex items-end gap-2">
            <span className="text-2xl font-black text-teal-600 tracking-tight">
              {totalAllKwhBulan.toFixed(2)}
            </span>
            <span className="text-sm font-bold text-teal-600/60 mb-1">kWh/bulan</span>
          </div>
        </div>

        <div className="bg-slate-50 rounded-2xl p-4 mt-6">
          <h4 className="text-xs font-bold text-slate-600 uppercase tracking-wider mb-3">Detail Peralatan</h4>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-500">Peralatan Utama</span>
              <span className="font-semibold text-slate-700">{totalMain.toFixed(2)} kWh</span>
            </div>
            {uiState.alatLainAda && (
              <div className="flex justify-between text-sm">
                <span className="text-slate-500">Peralatan Lainnya</span>
                <span className="font-semibold text-slate-700">{totalAlatLain.toFixed(2)} kWh</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
