import request from '@/utils/request'

export interface ConfigTemplate {
  id: number
  name: string
  description: string
  config_data: Record<string, any>
  is_default: boolean
  created_at: string
  updated_at: string
}

export interface CreateTemplateRequest {
  name: string
  description: string
  config_data: Record<string, any>
  is_default?: boolean
}

export const getTemplates = () => {
  return request.get<any, ConfigTemplate[]>('/history/templates')
}

export const createTemplate = (data: CreateTemplateRequest) => {
  return request.post<any, ConfigTemplate>('/history/templates', data)
}

export const deleteTemplate = (templateId: number) => {
  return request.delete(`/history/templates/${templateId}`)
}
