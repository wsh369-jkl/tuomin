import request from '@/utils/request'
import type { TaskStatus } from '@/api/desensitize'

export interface ReviewEvidenceRef {
  quote: string
  start: number
  end: number
  page?: number | null
  block_type: string
  evidence_quality: string
}

export interface ReviewCardItem {
  title?: string | null
  label?: string | null
  value?: string | null
  severity?: string | null
  reason?: string | null
  evidence_refs: ReviewEvidenceRef[]
  lawyer_action_hint?: string | null
}

export interface ReviewCard {
  type: string
  title: string
  items: ReviewCardItem[]
}

export interface CanonicalGroup {
  group_id: string
  group_label: string
  primary_text: string
  entity_type?: string
  canonical_role?: string | null
  aliases: string[]
  mentions: number
  confirmed?: boolean
  needs_review?: boolean
  review_reasons?: string[]
}

export interface ReviewIssue {
  title: string
  severity?: string | null
  reason?: string | null
  evidence_refs: ReviewEvidenceRef[]
  lawyer_action_hint?: string | null
}

export interface ReviewMetadata {
  analysis_tier?: string
  ocr_mode?: string | null
  precision_summary?: string | null
  review_model?: string | null
  review_generation_mode?: string | null
  review_generation_label?: string | null
  review_strategy_key?: string | null
  review_strategy_label?: string | null
  review_strategy_description?: string | null
  review_budget_tier?: string | null
  review_budget_label?: string | null
  specialized_passes?: string[]
  definition_recall_enabled?: boolean
  residual_scan_enabled?: boolean
  review_card_count?: number
  review_evidence_count?: number
  review_queue_count?: number
}

export interface ReviewResult {
  review_id: string
  task_id: string
  document_type: string
  document_type_label: string
  summary: string
  cards: ReviewCard[]
  canonical_groups: CanonicalGroup[]
  suspected_misses: ReviewIssue[]
  metadata: ReviewMetadata
}

export const uploadForReview = (
  file: File,
  useLlm = true,
  useCustom = true,
  llmModel?: string,
  anonymizationStrategy?: string
) => {
  const formData = new FormData()
  formData.append('file', file)

  return request.post<any, TaskStatus>('/review/upload', formData, {
    params: {
      use_llm: useLlm,
      use_custom: useCustom,
      llm_model: llmModel,
      anonymization_strategy: anonymizationStrategy,
      async_mode: true
    },
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

export const generateReview = (taskId: string, llmModel?: string) => {
  return request.post<any, TaskStatus>('/review/generate', {
    task_id: taskId,
    llm_model: llmModel
  })
}

export const getReviewStatus = (reviewId: string) => {
  return request.get<any, TaskStatus>(`/review/status/${reviewId}`)
}

export const getReviewResult = (reviewId: string) => {
  return request.get<any, ReviewResult>(`/review/result/${reviewId}`)
}
