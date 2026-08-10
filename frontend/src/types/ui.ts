export interface ApplianceState {
  jumlah: number;
  watt: number;
  jam: number;
  isCustomWatt?: boolean;
}

export interface UIState {
  // Profil Rumah
  anggotaKeluarga: string | number;
  dayaRumahVA: string | number;
  statusSubsidi: string; // '0' | '1' | ''
  
  // Khusus Prabayar
  nominalTokenTerakhir: string | number;
  frekuensiIsiToken: string | number;
  tokenNominalKategori: string;
  mesinCuciKategori: string;
  
  // Khusus Pascabayar
  tagihanStabil: string;

  // Peralatan
  kulkas: ApplianceState;
  tv: ApplianceState;
  ac: ApplianceState;
  kipas: ApplianceState;
  ricecooker: ApplianceState;
  mesincuci: ApplianceState & {
    frekuensiPerMinggu: number;
    kategori: string; // '0' | '1' | '2' | ''
  };

  // Alat Lain
  alatLainAda: boolean;
  alatLainTotalWatt: number;
  alatLainTotalJam: number;
}
