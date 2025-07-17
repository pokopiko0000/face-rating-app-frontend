import { useState, useCallback } from 'react';

export interface ValidationRule {
  required?: boolean;
  minLength?: number;
  maxLength?: number;
  pattern?: RegExp;
  custom?: (value: string) => string | null;
}

export interface ValidationRules {
  [key: string]: ValidationRule;
}

export interface ValidationErrors {
  [key: string]: string;
}

export function useFormValidation(rules: ValidationRules) {
  const [errors, setErrors] = useState<ValidationErrors>({});

  const validateField = useCallback((name: string, value: string): string | null => {
    const rule = rules[name];
    if (!rule) {
      return null;
    }

    // Required validation
    if (rule.required && !value.trim()) {
      return 'この項目は必須です';
    }

    // Skip other validations if field is empty and not required
    if (!value.trim() && !rule.required) {
      return null;
    }

    // Min length validation
    if (rule.minLength && value.length < rule.minLength) {
      return `${rule.minLength}文字以上で入力してください`;
    }

    // Max length validation
    if (rule.maxLength && value.length > rule.maxLength) {
      return `${rule.maxLength}文字以内で入力してください`;
    }

    // Pattern validation
    if (rule.pattern && !rule.pattern.test(value)) {
      return '入力形式が正しくありません';
    }

    // Custom validation
    if (rule.custom) {
      return rule.custom(value);
    }

    return null;
  }, [rules]);

  const validateForm = useCallback((formData: Record<string, string>) => {
    const newErrors: ValidationErrors = {};
    let isValid = true;

    Object.keys(rules).forEach(fieldName => {
      const error = validateField(fieldName, formData[fieldName] || '');
      if (error) {
        newErrors[fieldName] = error;
        isValid = false;
      }
    });

    setErrors(newErrors);
    return isValid;
  }, [rules, validateField]);

  const validateSingleField = useCallback((name: string, value: string) => {
    const error = validateField(name, value);
    setErrors(prev => ({
      ...prev,
      [name]: error || ''
    }));
    return !error;
  }, [validateField]);

  const clearErrors = useCallback(() => {
    setErrors({});
  }, []);

  const clearFieldError = useCallback((name: string) => {
    setErrors(prev => ({
      ...prev,
      [name]: ''
    }));
  }, []);

  return {
    errors,
    validateForm,
    validateSingleField,
    clearErrors,
    clearFieldError
  };
}