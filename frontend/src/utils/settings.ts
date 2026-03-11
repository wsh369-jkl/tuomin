export interface OperatorConfigItem {
  operator: string
  params?: Record<string, any>
}

export interface AppSettings {
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

export const defaultAppSettings: AppSettings = {
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
  return {
    use_llm_default: input.use_llm_default ?? defaultAppSettings.use_llm_default,
    llm_model_default: FIXED_LLM_MODEL,
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
