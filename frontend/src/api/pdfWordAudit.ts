import request from '@/utils/request'

export interface PdfWordAuditStatus {
  audit_id: string
  filename?: string | null
  template_filename?: string | null
  status: string
  progress: number
  message?: string | null
  error_message?: string | null
  created_at: string
}

export interface PdfWordAuditCorrection {
  id: string
  wps_unit_id: string
  page_no?: number | null
  old_text: string
  new_text: string
  action: string
  confidence: number
  alignment_score: number
  reason: string
  comment_text: string
  sensitive_low_priority: boolean
}

export interface PdfWordAuditFinding {
  id: string
  severity: string
  category: string
  page_no?: number | null
  wps_text: string
  suggested_text: string
  diff_ops: Record<string, any>[]
  confidence: number
  status: string
  evidence_sources: string[]
  bbox_refs: Record<string, any>[]
  crop_refs: string[]
  wps_anchor: Record<string, any>
  reason: string
  requires_human_review: boolean
}

export interface PdfWordAuditResult {
  audit_id: string
  filename: string
  template_filename: string
  status: string
  metadata: Record<string, any>
  product_report?: Record<string, any>
  page_risk_summary?: Record<string, any>[]
  table_summary?: Record<string, any>
  coverage_summary?: Record<string, any>
  artifact_manifest?: Record<string, any>
  review_task_summary?: Record<string, any>
  human_review_queue?: Record<string, any>[]
  findings: PdfWordAuditFinding[]
  corrections: PdfWordAuditCorrection[]
  download_url?: string | null
  report_url?: string | null
  evidence_url?: string | null
  output_filename?: string | null
}

export const uploadForPdfWordAudit = (pdfFile: File, wpsDocxFile: File) => {
  const formData = new FormData()
  formData.append('pdf_file', pdfFile)
  formData.append('wps_docx_file', wpsDocxFile)
  return request.post<any, PdfWordAuditStatus>('/pdf-word-audit/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  })
}

export const getPdfWordAuditStatus = (auditId: string) => {
  return request.get<any, PdfWordAuditStatus>(`/pdf-word-audit/status/${auditId}`)
}

export const getPdfWordAuditResult = (auditId: string) => {
  return request.get<any, PdfWordAuditResult>(`/pdf-word-audit/result/${auditId}`)
}

export const downloadPdfWordAuditResult = (auditId: string) =>
  `/api/v1/pdf-word-audit/download/${auditId}`

export const downloadPdfWordAuditReport = (auditId: string) =>
  `/api/v1/pdf-word-audit/report/${auditId}`

export const downloadPdfWordAuditEvidence = (auditId: string) =>
  `/api/v1/pdf-word-audit/evidence/${auditId}`
