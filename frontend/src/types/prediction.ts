export type ModelType = "prepaid" | "postpaid";

export interface FieldConfig {
  name: string;
  label: string;
  type: "number" | "select" | "text" | "toggle";
  section: string;
  required: boolean;
  min?: number;
  max?: number;
  step?: number;
  placeholder?: string;
  helperText?: string;
  options?: { value: string; label: string }[];
  conditionalOn?: string;
  conditionalValue?: unknown;
}

export interface SectionConfig {
  id: string;
  title: string;
}

export type FormData = Record<string, string | number | boolean>;

export interface PredictionResponse {
  success: boolean;
  prediction: number;
  error?: string;
}

export interface ValidationErrors {
  [fieldName: string]: string;
}
