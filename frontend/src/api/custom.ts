import request from '@/utils/request'

export interface KeywordRule {
  description?: string
  keywords: string[]
  score: number
}

export interface PatternRule {
  name: string
  description?: string
  regex: string
  score: number
  context?: string[]
}

export interface CustomConfig {
  keywords: Record<string, KeywordRule>
  patterns: PatternRule[]
  keywords_count: number
  patterns_count: number
}

export interface AddKeywordRequest {
  entity_type: string
  keywords: string[]
  score: number
  description?: string
}

export interface AddPatternRequest {
  entity_type: string
  regex: string
  context: string[]
  score: number
  description?: string
}

export interface TestRecognizerResponse {
  text: string
  count: number
  entities: Array<{
    type: string
    text: string
    start: number
    end: number
    score: number
    source: string
    metadata?: Record<string, unknown>
  }>
}

export const getCustomConfig = () => {
  return request.get<any, CustomConfig>('/custom/config')
}

export const addKeywords = (data: AddKeywordRequest) => {
  return request.post('/custom/keywords', data)
}

export const addPattern = (data: AddPatternRequest) => {
  return request.post('/custom/patterns', data)
}

export const deleteKeywords = (entityType: string) => {
  return request.delete(`/custom/keywords/${entityType}`)
}

export const deletePattern = (entityType: string) => {
  return request.delete(`/custom/patterns/${entityType}`)
}

export const reloadConfig = () => {
  return request.post('/custom/reload')
}

export const testRecognizer = (text: string) => {
  return request.get<any, TestRecognizerResponse>('/custom/test', {
    params: { text }
  })
}
