import { computed, ref } from 'vue'
import { ElMessage } from 'element-plus'
import {
  getAvailableModels,
  getRuntimeStatus,
  type LLMModelListResponse,
  type LLMModelOption,
  type RuntimeStatusResponse
} from '@/api/desensitize'

export const useRuntimeModels = () => {
  const modelCatalog = ref<LLMModelListResponse | null>(null)
  const runtimeStatus = ref<RuntimeStatusResponse | null>(null)

  const isHighQualityLowmem = computed(
    () =>
      modelCatalog.value?.profile === 'high_quality_lowmem' ||
      runtimeStatus.value?.desensitize_mode === 'high_quality_lowmem'
  )
  const isHighQualityWorkflow = computed(() => isHighQualityLowmem.value)
  const allModelOptions = computed(() => modelCatalog.value?.models || [])
  const llmModelOptions = computed(() => {
    const models = allModelOptions.value
    if (!isHighQualityWorkflow.value) {
      return models
    }
    return models.filter((item) =>
      ['review', 'review_fallback'].includes(item.role || item.tier)
    )
  })
  const runtimeReady = computed(() => runtimeStatus.value?.ready ?? false)

  const syncSelectedModel = (
    preferredModel?: string | null,
    options?: {
      requireInstalled?: boolean
      tier?: string
    }
  ) => {
    const requireInstalled = options?.requireInstalled ?? true
    const desiredTier = options?.tier
    const filteredOptions = llmModelOptions.value.filter((item) => {
      const installedPass = !requireInstalled || item.installed
      const tierPass =
        !desiredTier ||
        item.tier === desiredTier ||
            (isHighQualityWorkflow.value &&
              desiredTier === 'review' &&
              ['review', 'review_fallback'].includes(item.role || item.tier))
      return installedPass && tierPass
    })
    const optionNames = filteredOptions.map((item) => item.name)

    if (preferredModel && optionNames.includes(preferredModel)) {
      return preferredModel
    }

    const defaultModel = desiredTier
      ? filteredOptions.find((item) => item.is_default || item.tier === desiredTier)?.name
      : modelCatalog.value?.default_model
    if (defaultModel && optionNames.includes(defaultModel)) {
      return defaultModel
    }

    return optionNames[0] || ''
  }

  const formatModelLabel = (model: LLMModelOption) => {
    const status = model.installed ? '已安装' : '未安装'
    const defaultTag = model.is_default ? ' / 默认' : ''
    const roleKey = model.role || model.tier
    const roleMap: Record<string, string> = {
      primary_ie: '中文实体模型',
      primary_ner: '中文 NER',
      secondary_ner: '第二路 NER',
      review: '按需精审',
      review_fallback: '低内存兜底精审'
    }
    const roleTag = isHighQualityWorkflow.value && roleMap[roleKey] ? ` / ${roleMap[roleKey]}` : ''
    const capabilityTag = !isHighQualityWorkflow.value && model.tier === 'review' ? ' / 精审' : ''
    return `${model.name} (${model.strategy_label}${roleTag}${capabilityTag} / ${status}${defaultTag})`
  }

  const loadModelCatalog = async (desensitizeMode?: string) => {
    try {
      modelCatalog.value = await getAvailableModels(desensitizeMode)
    } catch (error) {
      ElMessage.error('加载模型列表失败')
    }
  }

  const loadRuntimeStatus = async (desensitizeMode?: string) => {
    try {
      runtimeStatus.value = await getRuntimeStatus(desensitizeMode)
    } catch (error) {
      ElMessage.error('读取运行环境状态失败')
    }
  }

  return {
    modelCatalog,
    runtimeStatus,
    isHighQualityLowmem,
    isHighQualityWorkflow,
    allModelOptions,
    llmModelOptions,
    runtimeReady,
    syncSelectedModel,
    formatModelLabel,
    loadModelCatalog,
    loadRuntimeStatus
  }
}
