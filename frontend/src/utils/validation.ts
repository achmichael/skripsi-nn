import type { FieldConfig, FormData, ValidationErrors } from "../types/prediction";

export function validateField(field: FieldConfig, value: unknown, formData: FormData): string | null {
  if (field.conditionalOn) {
    const depVal = formData[field.conditionalOn];
    if (depVal !== field.conditionalValue) return null;
  }

  if (field.required) {
    if (value === "" || value === undefined || value === null) {
      return `${field.label} harus diisi.`;
    }
  }

  if (field.type === "number" && value !== "" && value !== undefined) {
    const num = Number(value);
    if (isNaN(num)) return `${field.label} harus berupa angka.`;
    if (field.min !== undefined && num < field.min) return `${field.label} minimal ${field.min}.`;
    if (field.max !== undefined && num > field.max) return `${field.label} maksimal ${field.max}.`;
  }

  return null;
}

export function validateAllFields(fields: FieldConfig[], formData: FormData): ValidationErrors {
  const errors: ValidationErrors = {};
  for (const field of fields) {
    const err = validateField(field, formData[field.name], formData);
    if (err) errors[field.name] = err;
  }
  return errors;
}
