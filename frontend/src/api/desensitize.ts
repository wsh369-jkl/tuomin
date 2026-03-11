import request from '@/utils/request'

export interface Entity {
  type: string
  text: string
  start: number
  end: number
  score: number
  source: string
  replacement?: string
  replacement_method?: string
  context_label?: string
  context_role?: string
  canonical_key?: string
  canonical_role?: string
}

export interface AnalyzeResponse {
  task_id: string
  filename: string
  text: string
  entities: Entity[]
  statistics: Record<string, any>
  metadata: Record<string, any>
  llm_model?: string
  llm_strategy?: string
  llm_strategy_label?: string
  anonymization_strategy?: string
  anonymization_strategy_label?: string
}

export interface DesensitizeRequest {
  task_id: string
  entities: Entity[]
  config?: Record<string, any>
  llm_model?: string
  anonymization_strategy?: string
}

export interface DesensitizeResponse {
  task_id: string
  status: string
  anonymized_text?: string
  entities: Entity[]
  metadata?: Record<string, any>
  download_url?: string
  output_filename?: string
  output_file_type?: string
  preserves_format?: boolean
  llm_assisted?: boolean
  llm_model?: string
  llm_strategy?: string
  llm_strategy_label?: string
  anonymization_strategy?: string
  anonymization_strategy_label?: string
  warning?: string
  message?: string
}

export interface EngineInfo {
  version: string
  architecture: string
  llm_backend: string
  llm_model: string
  supported_entities: string[]
  supported_operators: string[]
  statistics: Record<string, any>
}

export interface LLMModelOption {
  name: string
  installed: boolean
  is_default: boolean
  strategy_key: string
  strategy_label: string
  strategy_description: string
}

export interface LLMModelListResponse {
  backend: string
  default_model?: string
  service_available: boolean
  models: LLMModelOption[]
}

export interface RuntimeStatusResponse {
  backend: string
  platform: string
  ready: boolean
  ollama_install_detected: boolean
  ollama_path?: string
  service_available: boolean
  required_model?: string
  required_model_installed: boolean
  default_model?: string
  installer_hint: string
  download_hint: string
  recommended_action: string
}

export const uploadAndAnalyze = (
  file: File,
  useLlm = true,
  useCustom = true,
  llmModel?: string,
  anonymizationStrategy?: string
) => {
  const formData = new FormData()
  formData.append('file', file)

  return request.post<any, AnalyzeResponse>('/desensitize/upload', formData, {
    params: {
      use_llm: useLlm,
      use_custom: useCustom,
      llm_model: llmModel,
      anonymization_strategy: anonymizationStrategy
    },
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

export const processDesensitize = (data: DesensitizeRequest) => {
  return request.post<any, DesensitizeResponse>('/desensitize/process', data)
}

export const downloadResult = (taskId: string) => `/api/v1/desensitize/download/${taskId}`

export const getTaskStatus = (taskId: string) => {
  return request.get(`/desensitize/status/${taskId}`)
}

export const getEngineInfo = () => {
  return request.get<any, EngineInfo>('/desensitize/info')
}

export const getAvailableModels = () => {
  return request.get<any, LLMModelListResponse>('/desensitize/models')
}

export const getRuntimeStatus = () => {
  return request.get<any, RuntimeStatusResponse>('/desensitize/runtime-status')
}
