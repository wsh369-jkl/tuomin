import request from '@/utils/request'

export interface AssistantEvidenceRef {
  quote: string
  start: number
  end: number
  page?: number | null
  block_type: string
  evidence_quality: string
}

export interface AssistantItem {
  title?: string | null
  label?: string | null
  value?: string | null
  status?: string | null
  severity?: string | null
  reason?: string | null
  evidence_refs: AssistantEvidenceRef[]
  action_hint?: string | null
}

export interface AssistantSection {
  type: string
  title: string
  items: AssistantItem[]
}

export interface AssistantTaskStatus {
  assistant_id: string
  filename?: string | null
  status: string
  progress: number
  stage_key?: string | null
  stage_label?: string | null
  message?: string | null
  error_message?: string | null
  created_at: string
}

export interface AssistantMetadata {
  assistant_model?: string | null
  ocr_mode?: string | null
  classification_stage?: string | null
  evidence_count?: number
  limited_reason?: string | null
  stage_trace?: Array<Record<string, any>>
  llm_document_type_reason?: string | null
  llm_document_type_confidence?: number | null
}

export interface AssistantResult {
  assistant_id: string
  filename: string
  document_type: string
  document_type_label: string
  support_mode: 'supported' | 'limited'
  support_notice: string
  summary: string
  sections: AssistantSection[]
  text: string
  metadata: AssistantMetadata
}

export const uploadForAssistant = (file: File, llmModel?: string) => {
  const formData = new FormData()
  formData.append('file', file)

  return request.post<any, AssistantTaskStatus>('/assistant/upload', formData, {
    params: {
      llm_model: llmModel
    },
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

export const getAssistantStatus = (assistantId: string) => {
  return request.get<any, AssistantTaskStatus>(`/assistant/status/${assistantId}`)
}

export const getAssistantResult = (assistantId: string) => {
  return request.get<any, AssistantResult>(`/assistant/result/${assistantId}`)
}
