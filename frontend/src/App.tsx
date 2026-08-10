import { useState, useCallback, useMemo } from "react";
import type {
  ModelType,
  FormData,
  PredictionResponse,
  ValidationErrors,
  FieldConfig,
} from "./types/prediction";
import { prepaidFields, prepaidSections } from "./data/prepaidFields";
import { postpaidFields, postpaidSections } from "./data/postpaidFields";
import { predictPrepaid, predictPostpaid } from "./services/predictionApi";
import Header from "./components/Header";
import ModelSelector from "./components/ModelSelector";
import FormSection from "./components/FormSection";
import FormField from "./components/FormField";
import { validateAllFields } from "./utils/validation";
import PredictionActions from "./components/PredictionActions";
import ResultCard from "./components/ResultCard";

function buildDefaults(fields: FieldConfig[]): FormData {
  const data: FormData = {};
  for (const f of fields) {
    if (f.type === "toggle") data[f.name] = false;
    else data[f.name] = "";
  }
  return data;
}

// TODO: Replace local calculation with backend/model value if required.
function autoCalculate(data: FormData, fields: FieldConfig[]): FormData {
  const next = { ...data };
  const fieldNames = new Set(fields.map((f) => f.name));

  const calc = (prefix: string) => {
    const jumlah = Number(next[`${prefix}_Jumlah`]) || 0;
    const watt = Number(next[`${prefix}_EstimasiWattPerUnit`]) || 0;
    const jam = Number(next[`${prefix}_EstimasiJamPerHari`]) || 0;
    if (jumlah > 0 && watt > 0 && jam > 0) {
      const energi = (jumlah * watt * jam) / 1000;
      const key = `${prefix}_Energi_kWhPerHari`;
      if (fieldNames.has(key) && (next[key] === "" || next[key] === 0)) {
        next[key] = Math.round(energi * 100) / 100;
      }
    }
  };

  calc("Kulkas");
  calc("TV");
  calc("AC");
  calc("Kipas");
  calc("RiceCooker");

  const mcJumlah = Number(next["MesinCuci_Jumlah"]) || 0;
  const mcWatt = Number(next["MesinCuci_EstimasiWattPerUnit"]) || 0;
  const mcFreq = Number(next["MesinCuci_EstimasiFrekuensiPerMinggu"]) || 0;
  const mcDurasi = Number(next["MesinCuci_EstimasiDurasiSekaliPakaiJam"]) || 0;
  if (mcJumlah > 0 && mcFreq > 0) {
    const watt = mcWatt > 0 ? mcWatt : 0;
    const durasi = mcDurasi > 0 ? mcDurasi : 1;
    const energi = (mcJumlah * watt * mcFreq * durasi) / (7 * 1000);
    const key = "MesinCuci_Energi_kWhPerHari";
    if (fieldNames.has(key) && (next[key] === "" || next[key] === 0)) {
      next[key] = Math.round(energi * 100) / 100;
    }
  }

  const mainDevices = [
    "Kulkas",
    "TV",
    "AC",
    "Kipas",
    "RiceCooker",
    "MesinCuci",
  ];
  let totalMain = 0;
  for (const d of mainDevices) {
    totalMain += Number(next[`${d}_Energi_kWhPerHari`]) || 0;
  }

  let totalAlatLain = 0;
  for (let i = 1; i <= 3; i++) {
    const w = Number(next[`Alat_Lain_${i}_EstimasiWatt`]) || 0;
    const j = Number(next[`Alat_Lain_${i}_Jumlah`]) || 0;
    if (w > 0 && j > 0) totalAlatLain += (j * w * 2) / 1000;
  }

  if (
    fieldNames.has("Total_Energi_Alat_Lain_kWhPerHari") &&
    (next["Total_Energi_Alat_Lain_kWhPerHari"] === "" ||
      next["Total_Energi_Alat_Lain_kWhPerHari"] === 0)
  ) {
    next["Total_Energi_Alat_Lain_kWhPerHari"] =
      Math.round(totalAlatLain * 100) / 100;
  }
  if (
    fieldNames.has("Total_Energi_Utama_kWhPerHari") &&
    (next["Total_Energi_Utama_kWhPerHari"] === "" ||
      next["Total_Energi_Utama_kWhPerHari"] === 0)
  ) {
    next["Total_Energi_Utama_kWhPerHari"] = Math.round(totalMain * 100) / 100;
  }

  const totalAll = totalMain + totalAlatLain;
  if (
    fieldNames.has("Total_Energi_Semua_kWhPerHari") &&
    (next["Total_Energi_Semua_kWhPerHari"] === "" ||
      next["Total_Energi_Semua_kWhPerHari"] === 0)
  ) {
    next["Total_Energi_Semua_kWhPerHari"] = Math.round(totalAll * 100) / 100;
  }
  if (
    fieldNames.has("Total_Energi_Semua_kWhPerBulan") &&
    (next["Total_Energi_Semua_kWhPerBulan"] === "" ||
      next["Total_Energi_Semua_kWhPerBulan"] === 0)
  ) {
    next["Total_Energi_Semua_kWhPerBulan"] =
      Math.round(totalAll * 30 * 100) / 100;
  }

  return next;
}

export default function App() {
  const [modelType, setModelType] = useState<ModelType>("prepaid");
  const [prepaidData, setPrepaidData] = useState<FormData>(() =>
    buildDefaults(prepaidFields),
  );
  const [postpaidData, setPostpaidData] = useState<FormData>(() =>
    buildDefaults(postpaidFields),
  );
  const [errors, setErrors] = useState<ValidationErrors>({});
  const [isLoading, setIsLoading] = useState(false);
  const [predictionResult, setPredictionResult] =
    useState<PredictionResponse | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [openSections, setOpenSections] = useState<Set<string>>(
    new Set(["customer"]),
  );

  const fields = modelType === "prepaid" ? prepaidFields : postpaidFields;
  const sections = modelType === "prepaid" ? prepaidSections : postpaidSections;
  const formData = modelType === "prepaid" ? prepaidData : postpaidData;
  const setFormData =
    modelType === "prepaid" ? setPrepaidData : setPostpaidData;

  const handleModelChange = useCallback((type: ModelType) => {
    setModelType(type);
    setErrors({});
    setPredictionResult(null);
    setPredictionError(null);
    setOpenSections(new Set(["customer"]));
  }, []);

  const handleFieldChange = useCallback(
    (name: string, value: string | number | boolean) => {
      setFormData((prev) => ({ ...prev, [name]: value }));
      setErrors((prev) => {
        if (!prev[name]) return prev;
        const next = { ...prev };
        delete next[name];
        return next;
      });
    },
    [setFormData],
  );

  const toggleSection = useCallback((id: string) => {
    setOpenSections((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const filledCount = useMemo(() => {
    let count = 0;
    for (const f of fields) {
      const v = formData[f.name];
      if (v !== "" && v !== undefined && v !== null && v !== false) count++;
    }
    return count;
  }, [fields, formData]);

  const sectionFieldCounts = useMemo(() => {
    const counts: Record<string, { total: number; filled: number }> = {};
    for (const s of sections) {
      counts[s.id] = { total: 0, filled: 0 };
    }
    for (const f of fields) {
      if (f.conditionalOn) {
        const depVal = formData[f.conditionalOn];
        if (depVal !== f.conditionalValue) continue;
      }
      if (counts[f.section]) {
        counts[f.section].total++;
        const v = formData[f.name];
        if (v !== "" && v !== undefined && v !== null && v !== false) {
          counts[f.section].filled++;
        }
      }
    }
    return counts;
  }, [fields, sections, formData]);

  const handlePredict = useCallback(async () => {
    const calculated = autoCalculate(formData, fields);
    setFormData(() => calculated);

    const validationErrors = validateAllFields(fields, calculated);
    if (Object.keys(validationErrors).length > 0) {
      setErrors(validationErrors);
      const firstErrorField = fields.find((f) => validationErrors[f.name]);
      if (firstErrorField) {
        setOpenSections((prev) => new Set([...prev, firstErrorField.section]));
        setTimeout(() => {
          document
            .getElementById(firstErrorField.name)
            ?.scrollIntoView({ behavior: "smooth", block: "center" });
        }, 200);
      }
      return;
    }

    setIsLoading(true);
    setPredictionResult(null);
    setPredictionError(null);

    try {
      const payload: FormData = {};
      for (const f of fields) {
        payload[f.name] = calculated[f.name];
      }

      const result =
        modelType === "prepaid"
          ? await predictPrepaid(payload)
          : await predictPostpaid(payload);

      setPredictionResult(result);
      setTimeout(() => {
        document
          .getElementById("result-card")
          ?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 100);
    } catch (err) {
      console.error("Prediction failed:", err);
      setPredictionError("Terjadi kesalahan saat memproses prediksi.");
    } finally {
      setIsLoading(false);
    }
  }, [formData, fields, modelType, setFormData]);

  const handleReset = useCallback(() => {
    const hasFilledData = filledCount > 3;
    if (hasFilledData && !window.confirm("Reset semua data form?")) return;

    setFormData(() => buildDefaults(fields));
    setErrors({});
    setPredictionResult(null);
    setPredictionError(null);
    setOpenSections(new Set(["customer"]));
  }, [fields, filledCount, setFormData]);

  const handleRetry = useCallback(() => {
    setPredictionError(null);
    handlePredict();
  }, [handlePredict]);

  const visibleFieldsForSection = useCallback(
    (sectionId: string) => {
      return fields.filter((f) => {
        if (f.section !== sectionId) return false;
        if (f.conditionalOn) {
          return formData[f.conditionalOn] === f.conditionalValue;
        }
        return true;
      });
    },
    [fields, formData],
  );

  return (
    <div className="min-h-screen bg-white">
      <Header />

      <div className="mx-auto max-w-6xl px-5 sm:px-8">
        {/* hero */}
        <div className="pb-8 pt-10 sm:pt-14">
          <p className="font-mono text-[11px] tracking-[0.2em] text-copper">
            FORM PENGUKURAN
          </p>
          <h1 className="mt-1 font-display text-3xl font-bold tracking-tight text-ink sm:text-4xl">
            Prediksi Konsumsi Listrik
          </h1>
          <p className="mt-2 max-w-lg text-[14px] leading-relaxed text-muted">
            Masukkan data pelanggan dan peralatan rumah tangga untuk mendapatkan
            estimasi penggunaan listrik.
          </p>
        </div>

        <ModelSelector modelType={modelType} onChange={handleModelChange} />

        <div className="mt-8 flex flex-col gap-10 lg:flex-row">
          <div className="min-w-0 flex-1 rounded-lg border border-line bg-panel px-5">
            {sections.map((section, i) => {
              const sectionFields = visibleFieldsForSection(section.id);
              const counts = sectionFieldCounts[section.id] ?? {
                total: 0,
                filled: 0,
              };
              return (
                <FormSection
                  key={section.id}
                  id={section.id}
                  title={section.title}
                  index={i + 1}
                  fieldCount={counts.total}
                  filledCount={counts.filled}
                  isOpen={openSections.has(section.id)}
                  onToggle={() => toggleSection(section.id)}
                >
                  {sectionFields.map((field) => (
                    <FormField
                      key={field.name}
                      field={field}
                      value={formData[field.name] ?? ""}
                      error={errors[field.name]}
                      onChange={handleFieldChange}
                    />
                  ))}
                </FormSection>
              );
            })}
            <div className="pb-6 pt-6">
              <PredictionActions
                isLoading={isLoading}
                onPredict={handlePredict}
                onReset={handleReset}
              />
            </div>
          </div>

          <div className="w-full shrink-0 lg:w-[280px]">
            <div className="lg:sticky lg:top-6">
              <ResultCard
                modelType={modelType}
                result={predictionResult}
                error={predictionError}
                onRetry={handleRetry}
                filledCount={filledCount}
                totalCount={fields.length}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
