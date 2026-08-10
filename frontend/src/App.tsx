import { useState, useCallback } from "react";
import type { ModelType, PredictionResponse } from "./types/prediction";
import type { UIState } from "./types/ui";
import { predictPrepaid, predictPostpaid } from "./services/predictionApi";

import Header from "./components/Header";
import ModelSelector from "./components/ModelSelector";
import ResultCard from "./components/ResultCard";
import WizardStepper from "./components/WizardStepper";
import ApplianceCard from "./components/ApplianceCard";
import EnergySummarySidebar from "./components/EnergySummarySidebar";

import { 
  AC_PRESETS, KULKAS_PRESETS, TV_PRESETS, 
  KIPAS_PRESETS, RICECOOKER_PRESETS, MESINCUCI_PRESETS 
} from "./data/appliancePresets";

import { 
  Refrigerator, Tv, Snowflake, Fan, 
  Coffee, WashingMachine, ArrowRight, ArrowLeft 
} from "lucide-react"; 

const defaultAppliance = { jumlah: 0, watt: 0, jam: 0 };
const defaultMesinCuci = { ...defaultAppliance, frekuensiPerMinggu: 0, kategori: "" };

const getInitialUIState = (): UIState => ({
  anggotaKeluarga: "",
  dayaRumahVA: "",
  statusSubsidi: "",
  
  nominalTokenTerakhir: "",
  frekuensiIsiToken: "",
  tokenNominalKategori: "",
  mesinCuciKategori: "",
  
  tagihanStabil: "",

  kulkas: { ...defaultAppliance, isCustomWatt: false, watt: KULKAS_PRESETS[0].watt },
  tv: { ...defaultAppliance, isCustomWatt: false, watt: TV_PRESETS[0].watt },
  ac: { ...defaultAppliance, isCustomWatt: false, watt: AC_PRESETS[0].watt },
  kipas: { ...defaultAppliance, isCustomWatt: false, watt: KIPAS_PRESETS[0].watt },
  ricecooker: { ...defaultAppliance, isCustomWatt: false, watt: RICECOOKER_PRESETS[0].watt },
  mesincuci: { ...defaultMesinCuci, isCustomWatt: false, watt: MESINCUCI_PRESETS[0].watt },

  alatLainAda: false,
  alatLainTotalWatt: 0,
  alatLainTotalJam: 0,
});

export default function App() {
  const [modelType, setModelType] = useState<ModelType>("prepaid");
  const [uiState, setUiState] = useState<UIState>(getInitialUIState());
  
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState<PredictionResponse | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  const handleModelChange = useCallback((type: ModelType) => {
    setModelType(type);
    setPredictionResult(null);
    setPredictionError(null);
  }, []);

  const handleChange = (field: keyof UIState, value: any) => {
    setUiState(prev => ({ ...prev, [field]: value }));
  };

  const mapUIStateToPayload = (state: UIState, type: ModelType) => {
    const payload: Record<string, string | number | boolean> = {};

    const calcKwh = (jumlah: number, watt: number, jam: number) => {
      return (jumlah * watt * jam) / 1000;
    };
    
    const kulkasKwh = calcKwh(state.kulkas.jumlah, state.kulkas.watt, state.kulkas.jam);
    const tvKwh = calcKwh(state.tv.jumlah, state.tv.watt, state.tv.jam);
    const acKwh = calcKwh(state.ac.jumlah, state.ac.watt, state.ac.jam);
    const kipasKwh = calcKwh(state.kipas.jumlah, state.kipas.watt, state.kipas.jam);
    const ricecookerKwh = calcKwh(state.ricecooker.jumlah, state.ricecooker.watt, state.ricecooker.jam);
    
    let mesinCuciKwh = 0;
    if (state.mesincuci.jumlah > 0 && state.mesincuci.frekuensiPerMinggu > 0) {
      const watt = state.mesincuci.watt > 0 ? state.mesincuci.watt : 0;
      const durasi = state.mesincuci.jam > 0 ? state.mesincuci.jam : 1;
      mesinCuciKwh = (state.mesincuci.jumlah * watt * state.mesincuci.frekuensiPerMinggu * durasi) / (7 * 1000);
    }

    let alatLainKwh = 0;
    if (state.alatLainAda) {
      alatLainKwh = (state.alatLainTotalWatt * state.alatLainTotalJam) / 1000;
    }

    const totalUtama = kulkasKwh + tvKwh + acKwh + kipasKwh + ricecookerKwh + mesinCuciKwh;
    const totalSemua = totalUtama + alatLainKwh;

    payload["Jumlah_Anggota_Keluarga"] = Number(state.anggotaKeluarga) || 1;
    payload["Daya_Listrik_Rumah_VA"] = Number(state.dayaRumahVA) || 900;
    payload["Status_Subsidi_Listrik"] = state.statusSubsidi || "0";

    if (type === "prepaid") {
      payload["Nominal_Token_Terakhir_Rp"] = Number(state.nominalTokenTerakhir) || 0;
      payload["Frekuensi_Isi_Token_Per_Bulan"] = Number(state.frekuensiIsiToken) || 0;
      payload["Token_Nominal_Kategori"] = state.tokenNominalKategori || "0";

      payload["Kulkas_Jumlah"] = state.kulkas.jumlah;
      payload["Kulkas_EstimasiJamPerHari"] = state.kulkas.jam;
      payload["Kulkas_Energi_kWhPerHari"] = Math.round(kulkasKwh * 100) / 100;

      payload["TV_Jumlah"] = state.tv.jumlah;
      payload["TV_EstimasiJamPerHari"] = state.tv.jam;
      payload["TV_Energi_kWhPerHari"] = Math.round(tvKwh * 100) / 100;

      payload["AC_Jumlah"] = state.ac.jumlah;
      payload["AC_EstimasiJamPerHari"] = state.ac.jam;
      payload["AC_Energi_kWhPerHari"] = Math.round(acKwh * 100) / 100;

      payload["Kipas_Jumlah"] = state.kipas.jumlah;
      payload["Kipas_EstimasiJamPerHari"] = state.kipas.jam;
      payload["Kipas_Energi_kWhPerHari"] = Math.round(kipasKwh * 100) / 100;

      payload["RiceCooker_Jumlah"] = state.ricecooker.jumlah;
      payload["RiceCooker_EstimasiJamPerHari"] = state.ricecooker.jam;
      payload["RiceCooker_Energi_kWhPerHari"] = Math.round(ricecookerKwh * 100) / 100;

      payload["MesinCuci_Jumlah"] = state.mesincuci.jumlah;
      payload["MesinCuci_Kategori"] = state.mesincuci.kategori || "0";
      payload["MesinCuci_EstimasiWattPerUnit"] = state.mesincuci.watt;
      payload["MesinCuci_EstimasiFrekuensiPerMinggu"] = state.mesincuci.frekuensiPerMinggu;
      payload["MesinCuci_EstimasiDurasiSekaliPakaiJam"] = state.mesincuci.jam;
      payload["MesinCuci_Energi_kWhPerHari"] = Math.round(mesinCuciKwh * 100) / 100;

      payload["Alat_Lain_Ada"] = state.alatLainAda;
      payload["Total_Energi_Alat_Lain_kWhPerHari"] = Math.round(alatLainKwh * 100) / 100;

      payload["Total_Energi_Utama_kWhPerHari"] = Math.round(totalUtama * 100) / 100;
      payload["Total_Energi_Semua_kWhPerHari"] = Math.round(totalSemua * 100) / 100;
    } else {
      payload["Tagihan_Relatif_Stabil__Ya, relatif stabil"] = state.tagihanStabil || "0";

      payload["Kulkas_Jumlah"] = state.kulkas.jumlah;
      payload["Kulkas_EstimasiWattPerUnit"] = state.kulkas.watt;
      payload["Kulkas_EstimasiJamPerHari"] = state.kulkas.jam;
      payload["Kulkas_Energi_kWhPerHari"] = Math.round(kulkasKwh * 100) / 100;

      payload["TV_Jumlah"] = state.tv.jumlah;
      payload["TV_EstimasiJamPerHari"] = state.tv.jam;
      payload["TV_Energi_kWhPerHari"] = Math.round(tvKwh * 100) / 100;

      payload["AC_Jumlah"] = state.ac.jumlah;
      payload["AC_EstimasiWattPerUnit"] = state.ac.watt;
      payload["AC_EstimasiJamPerHari"] = state.ac.jam;
      payload["AC_Energi_kWhPerHari"] = Math.round(acKwh * 100) / 100;

      payload["Kipas_Jumlah"] = state.kipas.jumlah;
      payload["Kipas_EstimasiJamPerHari"] = state.kipas.jam;
      payload["Kipas_Energi_kWhPerHari"] = Math.round(kipasKwh * 100) / 100;

      payload["RiceCooker_Jumlah"] = state.ricecooker.jumlah;
      payload["RiceCooker_EstimasiJamPerHari"] = state.ricecooker.jam;
      payload["RiceCooker_Energi_kWhPerHari"] = Math.round(ricecookerKwh * 100) / 100;

      payload["MesinCuci_Jumlah"] = state.mesincuci.jumlah;
      payload["MesinCuci_EstimasiFrekuensiPerMinggu"] = state.mesincuci.frekuensiPerMinggu;
      payload["MesinCuci_Energi_kWhPerHari"] = Math.round(mesinCuciKwh * 100) / 100;

      payload["Alat_Lain_Ada"] = state.alatLainAda;
      payload["Total_Energi_Alat_Lain_kWhPerHari"] = Math.round(alatLainKwh * 100) / 100;

      payload["Total_Energi_Semua_kWhPerHari"] = Math.round(totalSemua * 100) / 100;
      payload["Total_Energi_Semua_kWhPerBulan"] = Math.round(totalSemua * 30 * 100) / 100;
      
      const daya = Number(state.dayaRumahVA);
      const isSubsidi = state.statusSubsidi === "1";
      const tarif = daya === 450 ? 415 : (daya <= 900 && isSubsidi ? 605 : (daya <= 900 ? 1352 : 1444));
      payload["Estimasi_Tarif_Per_kWh_Rp"] = tarif;
    }

    return payload;
  };

  const handlePredict = async () => {
    setIsLoading(true);
    setPredictionError(null);
    try {
      const payload = mapUIStateToPayload(uiState, modelType);
      const result = modelType === "prepaid" 
        ? await predictPrepaid(payload as any)
        : await predictPostpaid(payload as any);
        
      setPredictionResult(result);
      // Auto scroll to top on success
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      setPredictionError("Terjadi kesalahan saat memproses prediksi.");
    } finally {
      setIsLoading(false);
    }
  };

  // Views
  const renderStep1 = () => (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-500">
      <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-100 shadow-[0_8px_30px_rgb(0,0,0,0.04)]">
        <h2 className="text-xl font-bold text-slate-800 mb-6">Profil Rumah</h2>
        
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-700">Daya Listrik Rumah (VA)</label>
            <select 
              value={uiState.dayaRumahVA}
              onChange={(e) => handleChange("dayaRumahVA", e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-700 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
            >
              <option value="">Pilih Daya</option>
              <option value="450">450 VA</option>
              <option value="900">900 VA</option>
              <option value="1300">1300 VA</option>
              <option value="2200">2200 VA</option>
              <option value="3500">3500 VA</option>
              <option value="5500">5500 VA</option>
            </select>
          </div>
          
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-700">Status Subsidi</label>
            <select 
              value={uiState.statusSubsidi}
              onChange={(e) => handleChange("statusSubsidi", e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-700 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
            >
              <option value="">Pilih Status</option>
              <option value="0">Non-Subsidi</option>
              <option value="1">Subsidi</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-700">Jumlah Anggota Keluarga</label>
            <input 
              type="number" 
              min="1"
              value={uiState.anggotaKeluarga}
              onChange={(e) => handleChange("anggotaKeluarga", e.target.value)}
              placeholder="Contoh: 4"
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-700 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
            />
          </div>
        </div>
      </div>

      <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-100 shadow-[0_8px_30px_rgb(0,0,0,0.04)]">
        <h2 className="text-xl font-bold text-slate-800 mb-6">
          {modelType === "prepaid" ? "Riwayat Token Prabayar" : "Riwayat Tagihan Pascabayar"}
        </h2>
        
        {modelType === "prepaid" ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-700">Nominal Token Terakhir (Rp)</label>
              <input 
                type="number" 
                value={uiState.nominalTokenTerakhir}
                onChange={(e) => handleChange("nominalTokenTerakhir", e.target.value)}
                placeholder="Contoh: 100000"
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-700 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-700">Frekuensi Isi (Bulan)</label>
              <input 
                type="number" 
                value={uiState.frekuensiIsiToken}
                onChange={(e) => handleChange("frekuensiIsiToken", e.target.value)}
                placeholder="Berapa kali isi sebulan?"
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-700 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
              />
            </div>
            <div className="space-y-2 sm:col-span-2">
              <label className="text-sm font-semibold text-slate-700">Kategori Nominal Token</label>
              <select 
                value={uiState.tokenNominalKategori}
                onChange={(e) => handleChange("tokenNominalKategori", e.target.value)}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-700 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
              >
                <option value="">Pilih Kategori</option>
                <option value="0">Rendah (&lt; 50k)</option>
                <option value="1">Sedang (50k - 100k)</option>
                <option value="2">Tinggi (&gt; 100k)</option>
              </select>
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            <label className="text-sm font-semibold text-slate-700">Apakah tagihan relatif stabil?</label>
            <select 
              value={uiState.tagihanStabil}
              onChange={(e) => handleChange("tagihanStabil", e.target.value)}
              className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-700 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
            >
              <option value="">Pilih</option>
              <option value="1">Ya, Relatif Stabil</option>
              <option value="0">Tidak, Bervariasi</option>
            </select>
          </div>
        )}
      </div>
    </div>
  );

  const renderStep2 = () => (
    <div className="space-y-4 animate-in fade-in slide-in-from-right-4 duration-500">
      <ApplianceCard 
        title="Televisi" icon={<Tv className="w-6 h-6" />} state={uiState.tv} presets={TV_PRESETS}
        onChange={(val) => handleChange("tv", val)}
      />
      <ApplianceCard 
        title="Kulkas" icon={<Refrigerator className="w-6 h-6" />} state={uiState.kulkas} presets={KULKAS_PRESETS}
        onChange={(val) => handleChange("kulkas", val)}
      />
      <ApplianceCard 
        title="AC (Air Conditioner)" icon={<Snowflake className="w-6 h-6" />} state={uiState.ac} presets={AC_PRESETS}
        onChange={(val) => handleChange("ac", val)}
      />
      <ApplianceCard 
        title="Kipas Angin" icon={<Fan className="w-6 h-6" />} state={uiState.kipas} presets={KIPAS_PRESETS}
        onChange={(val) => handleChange("kipas", val)}
      />
      <ApplianceCard 
        title="Rice Cooker" icon={<Coffee className="w-6 h-6" />} state={uiState.ricecooker} presets={RICECOOKER_PRESETS}
        onChange={(val) => handleChange("ricecooker", val)}
      />
      
      {/* Mesin cuci butuh config tambahan */}
      <div className={`p-5 rounded-3xl border transition-all duration-300 ${uiState.mesincuci.jumlah > 0 ? 'bg-white border-teal-200 shadow-[0_8px_30px_rgb(0,0,0,0.04)] ring-1 ring-teal-50' : 'bg-slate-50/50 border-slate-200'}`}>
        <ApplianceCard 
          title="Mesin Cuci" icon={<WashingMachine className="w-6 h-6" />} state={uiState.mesincuci} presets={MESINCUCI_PRESETS}
          onChange={(val) => handleChange("mesincuci", val as any)}
        />
        {uiState.mesincuci.jumlah > 0 && (
          <div className="mt-4 pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-4 animate-in fade-in">
            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-700">Frekuensi (kali/minggu)</label>
              <input 
                type="number" value={uiState.mesincuci.frekuensiPerMinggu || ""}
                onChange={(e) => handleChange("mesincuci", { ...uiState.mesincuci, frekuensiPerMinggu: Number(e.target.value) })}
                className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-slate-700" placeholder="Misal: 3"
              />
            </div>
            {modelType === "prepaid" && (
              <div className="space-y-2">
                <label className="text-sm font-semibold text-slate-700">Kategori Mesin Cuci</label>
                <select 
                  value={uiState.mesincuci.kategori}
                  onChange={(e) => handleChange("mesincuci", { ...uiState.mesincuci, kategori: e.target.value })}
                  className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-2.5 text-slate-700"
                >
                  <option value="">Pilih</option>
                  <option value="0">Rendah</option>
                  <option value="1">Sedang</option>
                  <option value="2">Tinggi</option>
                </select>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );

  const renderStep3 = () => (
    <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-500">
      <div className="bg-white rounded-3xl p-6 sm:p-8 border border-slate-100 shadow-[0_8px_30px_rgb(0,0,0,0.04)]">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold text-slate-800">Peralatan Lainnya</h2>
            <p className="text-sm text-slate-500">Apakah ada alat listrik lain yang tidak disebutkan di step sebelumnya?</p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input type="checkbox" className="sr-only peer" checked={uiState.alatLainAda} onChange={(e) => handleChange("alatLainAda", e.target.checked)} />
            <div className="w-14 h-7 bg-slate-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-6 after:w-6 after:transition-all peer-checked:bg-teal-500"></div>
          </label>
        </div>

        {uiState.alatLainAda && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 pt-4 border-t border-slate-100 animate-in fade-in zoom-in-95 duration-300">
            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-700">Total Watt Semua Alat Lain</label>
              <div className="relative">
                <input 
                  type="number" value={uiState.alatLainTotalWatt || ""}
                  onChange={(e) => handleChange("alatLainTotalWatt", Number(e.target.value))}
                  placeholder="Misal: 500"
                  className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-700 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                />
                <span className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 font-medium">Watt</span>
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-sm font-semibold text-slate-700">Rata-rata Durasi/Hari</label>
              <div className="relative">
                <input 
                  type="number" value={uiState.alatLainTotalJam || ""}
                  onChange={(e) => handleChange("alatLainTotalJam", Number(e.target.value))}
                  placeholder="Misal: 4"
                  className="w-full bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 text-slate-700 focus:outline-none focus:ring-2 focus:ring-teal-500/20 focus:border-teal-500"
                />
                <span className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 font-medium">Jam</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {predictionResult && (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
          <ResultCard
            modelType={modelType}
            result={predictionResult}
            error={predictionError}
            onRetry={handlePredict}
            filledCount={10}
            totalCount={10}
          />
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-slate-50/50 pb-20 font-sans selection:bg-teal-500/30">
      <Header />

      <div className="mx-auto max-w-6xl px-5 sm:px-8">
        {/* hero */}
        <div className="pb-8 pt-10 sm:pt-14 text-center max-w-2xl mx-auto">
          <div className="inline-flex items-center justify-center px-3 py-1 rounded-full bg-teal-50 text-teal-600 font-medium text-xs tracking-wider mb-4 border border-teal-100">
            Kalkulator Cerdas
          </div>
          <h1 className="font-display text-4xl sm:text-5xl font-extrabold tracking-tight text-slate-900 mb-4">
            Prediksi Konsumsi Listrik
          </h1>
          <p className="text-[15px] sm:text-base leading-relaxed text-slate-500">
            Dapatkan estimasi akurat penggunaan energi rumah Anda. Pilih model langganan dan lengkapi data profil serta peralatan yang Anda gunakan.
          </p>
        </div>

        <div className="flex justify-center mb-10">
          <ModelSelector modelType={modelType} onChange={handleModelChange} />
        </div>

        <div className="flex flex-col lg:flex-row gap-8 lg:gap-12 relative">
          
          <div className="min-w-0 flex-1">
            <WizardStepper 
              currentStep={currentStep} 
              steps={["Profil Rumah", "Peralatan Utama", "Alat Lain & Hasil"]} 
            />

            <div className="mb-8">
              {currentStep === 1 && renderStep1()}
              {currentStep === 2 && renderStep2()}
              {currentStep === 3 && renderStep3()}
            </div>

            <div className="flex items-center justify-between pt-6 border-t border-slate-200/60">
              <button
                onClick={() => setCurrentStep(prev => Math.max(1, prev - 1))}
                className={`flex items-center gap-2 px-6 py-3 rounded-2xl font-bold text-sm transition-all
                  ${currentStep === 1 
                    ? 'opacity-0 invisible' 
                    : 'text-slate-500 hover:bg-slate-100 hover:text-slate-800'}`}
              >
                <ArrowLeft className="w-4 h-4" /> Kembali
              </button>
              
              {currentStep < 3 ? (
                <button
                  onClick={() => setCurrentStep(prev => Math.min(3, prev + 1))}
                  className="flex items-center gap-2 px-8 py-3 rounded-2xl font-bold text-sm bg-slate-900 text-white hover:bg-slate-800 shadow-md shadow-slate-900/20 transition-all hover:translate-x-1"
                >
                  Lanjut <ArrowRight className="w-4 h-4" />
                </button>
              ) : (
                <button
                  onClick={handlePredict}
                  disabled={isLoading}
                  className="flex items-center gap-2 px-8 py-3 rounded-2xl font-bold text-sm bg-teal-600 text-white hover:bg-teal-700 shadow-lg shadow-teal-500/30 transition-all hover:-translate-y-0.5 disabled:opacity-70 disabled:hover:translate-y-0"
                >
                  {isLoading ? 'Memproses...' : 'Lihat Prediksi'} 
                  {!isLoading && <ArrowRight className="w-4 h-4" />}
                </button>
              )}
            </div>
          </div>

          {/* Sticky Sidebar */}
          <div className="w-full lg:w-[320px] shrink-0">
            <div className="lg:sticky lg:top-8">
              <EnergySummarySidebar uiState={uiState} modelType={modelType} />
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}
