export interface OperatorConfigItem {
  operator: string
  params?: Record<string, any>
}

export type DesensitizeMode = 'high_quality_lowmem'

export interface AppSettings {
  desensitize_mode_default: DesensitizeMode
  use_llm_default: boolean
  llm_model_default: string
  use_custom_default: boolean
  anonymization_strategy_default: string
  operator_config: Record<string, OperatorConfigItem>
  template_id: number | null
  template_name: string | null
}

export const SETTINGS_STORAGE_KEY = 'settings'
export const FIXED_LLM_MODEL = 'qwen3.5:4b'
export const DEFAULT_DESENSITIZE_MODE: DesensitizeMode = 'high_quality_lowmem'

export const DESENSITIZE_MODE_LABEL = '高质量低内存'
export const DESENSITIZE_MODE_DESCRIPTION =
  'PDF 走 RapidOCR 低内存规范化，适合需要压低峰值内存的任务。'

export const normalizeDesensitizeMode = (value: unknown): DesensitizeMode => {
  const normalized = typeof value === 'string' ? value.trim().toLowerCase() : ''
  if (normalized === 'high_quality_lowmem' || normalized === 'lowmem') {
    return 'high_quality_lowmem'
  }
  return DEFAULT_DESENSITIZE_MODE
}

export const getDesensitizeModeLabel = (value: unknown) => {
  normalizeDesensitizeMode(value)
  return DESENSITIZE_MODE_LABEL
}

export const defaultAppSettings: AppSettings = {
  desensitize_mode_default: DEFAULT_DESENSITIZE_MODE,
  use_llm_default: true,
  llm_model_default: FIXED_LLM_MODEL,
  use_custom_default: true,
  anonymization_strategy_default: 'official',
  operator_config: {},
  template_id: null,
  template_name: null
}

export const normalizeAppSettings = (value: unknown): AppSettings => {
  if (!value || typeof value !== 'object') {
    return { ...defaultAppSettings }
  }

  const input = value as Partial<AppSettings>
  const normalizedMode = normalizeDesensitizeMode(input.desensitize_mode_default)
  const normalizedModel =
    typeof input.llm_model_default === 'string' && input.llm_model_default.trim()
      ? input.llm_model_default.trim()
      : defaultAppSettings.llm_model_default
  return {
    desensitize_mode_default: normalizedMode,
    use_llm_default: input.use_llm_default ?? defaultAppSettings.use_llm_default,
    llm_model_default: normalizedModel,
    use_custom_default: input.use_custom_default ?? defaultAppSettings.use_custom_default,
    anonymization_strategy_default:
      typeof input.anonymization_strategy_default === 'string' &&
      input.anonymization_strategy_default.trim()
        ? input.anonymization_strategy_default.trim()
        : defaultAppSettings.anonymization_strategy_default,
    operator_config:
      input.operator_config && typeof input.operator_config === 'object'
        ? input.operator_config
        : {},
    template_id: typeof input.template_id === 'number' ? input.template_id : null,
    template_name: typeof input.template_name === 'string' ? input.template_name : null
  }
}

export const loadAppSettings = (): AppSettings => {
  const saved = localStorage.getItem(SETTINGS_STORAGE_KEY)
  if (!saved) {
    return { ...defaultAppSettings }
  }

  try {
    return normalizeAppSettings(JSON.parse(saved))
  } catch (error) {
    console.error('Failed to parse saved settings', error)
    return { ...defaultAppSettings }
  }
}

export const saveAppSettings = (settings: AppSettings) => {
  localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings))
}
