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
  group_id?: string
  group_label?: string
  needs_review?: boolean
  review_reason?: string
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

export interface TaskStatus {
  task_id: string
  filename?: string
  status: string
  progress: number
  message?: string
  error_message?: string
  created_at: string
}

export interface BatchFileItem {
  item_id: string
  filename: string
  relative_path: string
  status: string
  progress: number
  message?: string
  error_message?: string
  entities_count: number
  output_filename?: string
  mapping_output_filename?: string
  output_file_type?: string
  mapping_output_file_type?: string
  preserves_format: boolean
  warning?: string
  download_url?: string
  mapping_download_url?: string
  metadata: Record<string, any>
}

export interface BatchTaskStatus {
  batch_id: string
  folder_name: string
  output_folder_name?: string
  status: string
  progress: number
  message?: string
  error_message?: string
  file_count: number
  completed_count: number
  succeeded_count: number
  failed_count: number
  created_at: string
  items: BatchFileItem[]
}

export interface BatchResult {
  batch_id: string
  folder_name: string
  output_folder_name?: string
  status: string
  progress: number
  message?: string
  error_message?: string
  file_count: number
  completed_count: number
  succeeded_count: number
  failed_count: number
  archive_download_url?: string
  archive_filename?: string
  items: BatchFileItem[]
}

export interface DesensitizeRequest {
  task_id: string
  entities: Entity[]
  config?: Record<string, any>
  llm_model?: string
  anonymization_strategy?: string
  desensitize_mode?: string
  page_session_id?: string
  async_mode?: boolean
}

export interface DesensitizeResponse {
  task_id: string
  status: string
  anonymized_text?: string
  entities: Entity[]
  metadata?: Record<string, any>
  download_url?: string
  mapping_download_url?: string
  output_filename?: string
  mapping_output_filename?: string
  output_file_type?: string
  mapping_output_file_type?: string
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
  desensitize_mode?: string
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
  tier: string
  supports_precision_review: boolean
  supports_vision: boolean
  recommended_for: string[]
  role?: string
  memory_tier?: string
  local_path?: string
}

export interface LLMModelListResponse {
  backend: string
  default_model?: string
  service_available: boolean
  models: LLMModelOption[]
  profile?: string
  primary_models?: string[]
  review_models?: string[]
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
  available_processing_models: string[]
  preferred_processing_model?: string
  default_model?: string
  installer_hint: string
  download_hint: string
  recommended_action: string
  desensitize_mode?: string
  primary_models_ready?: boolean
  review_model_installed?: boolean
  review_model_loaded?: boolean
  estimated_memory_tier?: string
  analysis_worker_process?: boolean
  analysis_stage_isolation?: boolean
  analysis_worker_timeout?: number
}

export const uploadAndAnalyze = (
  file: File,
  useLlm = true,
  useCustom = true,
  llmModel?: string,
  anonymizationStrategy?: string,
  desensitizeMode?: string,
  pageSessionId?: string,
  wpsDocxTemplate?: File | null
) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('use_llm', String(useLlm))
  formData.append('use_custom', String(useCustom))
  formData.append('async_mode', 'true')
  if (llmModel) {
    formData.append('llm_model', llmModel)
  }
  if (anonymizationStrategy) {
    formData.append('anonymization_strategy', anonymizationStrategy)
  }
  if (desensitizeMode) {
    formData.append('desensitize_mode', desensitizeMode)
  }
  if (pageSessionId) {
    formData.append('page_session_id', pageSessionId)
  }
  if (wpsDocxTemplate) {
    formData.append('wps_docx_template', wpsDocxTemplate)
  }

  return request.post<any, TaskStatus>('/desensitize/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

export const uploadAndProcessFolder = (
  files: File[],
  relativePaths: string[],
  folderName: string,
  useLlm = true,
  useCustom = true,
  llmModel?: string,
  anonymizationStrategy?: string,
  operatorConfig?: Record<string, any>,
  desensitizeMode?: string,
  pageSessionId?: string
) => {
  const formData = new FormData()
  files.forEach((file, index) => {
    formData.append('files', file, file.name)
    formData.append('relative_paths', relativePaths[index] || file.name)
  })
  formData.append('folder_name', folderName)
  formData.append('use_llm', String(useLlm))
  formData.append('use_custom', String(useCustom))
  if (llmModel) {
    formData.append('llm_model', llmModel)
  }
  if (anonymizationStrategy) {
    formData.append('anonymization_strategy', anonymizationStrategy)
  }
  if (operatorConfig && Object.keys(operatorConfig).length > 0) {
    formData.append('operator_config_json', JSON.stringify(operatorConfig))
  }
  if (desensitizeMode) {
    formData.append('desensitize_mode', desensitizeMode)
  }
  if (pageSessionId) {
    formData.append('page_session_id', pageSessionId)
  }

  return request.post<any, BatchTaskStatus>('/desensitize/batch/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

export const processDesensitize = (data: DesensitizeRequest) => {
  return request.post<any, DesensitizeResponse>('/desensitize/process', data)
}

export const processDesensitizeAsync = (data: DesensitizeRequest) => {
  return request.post<any, TaskStatus>('/desensitize/process', {
    ...data,
    async_mode: true
  })
}

export const getAnalyzeResult = (taskId: string) => {
  return request.get<any, AnalyzeResponse>(`/desensitize/result/${taskId}`)
}

export const getProcessedResult = (taskId: string) => {
  return request.get<any, DesensitizeResponse>(`/desensitize/processed-result/${taskId}`)
}

export const downloadResult = (taskId: string) => `/api/v1/desensitize/download/${taskId}`

export const downloadMappingResult = (taskId: string) =>
  `/api/v1/desensitize/download/mapping/${taskId}`

export const getTaskStatus = (taskId: string) => {
  return request.get<any, TaskStatus>(`/desensitize/status/${taskId}`)
}

export const getBatchTaskStatus = (batchId: string) => {
  return request.get<any, BatchTaskStatus>(`/desensitize/batch/status/${batchId}`)
}

export const getBatchResult = (batchId: string) => {
  return request.get<any, BatchResult>(`/desensitize/batch/result/${batchId}`)
}

export const downloadBatchArchive = (batchId: string) =>
  `/api/v1/desensitize/batch/download/${batchId}`

export const downloadBatchItem = (batchId: string, itemId: string) =>
  `/api/v1/desensitize/batch/download/${batchId}/${itemId}`

export const downloadBatchItemMapping = (batchId: string, itemId: string) =>
  `/api/v1/desensitize/batch/download/${batchId}/${itemId}/mapping`

export const getEngineInfo = (desensitizeMode?: string) => {
  return request.get<any, EngineInfo>('/desensitize/info', {
    params: { desensitize_mode: desensitizeMode }
  })
}

export const getAvailableModels = (desensitizeMode?: string) => {
  return request.get<any, LLMModelListResponse>('/desensitize/models', {
    params: { desensitize_mode: desensitizeMode }
  })
}

export const getRuntimeStatus = (desensitizeMode?: string) => {
  return request.get<any, RuntimeStatusResponse>('/desensitize/runtime-status', {
    params: { desensitize_mode: desensitizeMode }
  })
}
