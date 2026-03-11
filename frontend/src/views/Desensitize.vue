<template>
  <div class="desensitize-page">
    <el-row :gutter="20">
      <el-col :xs="24" :lg="8">
        <el-card class="panel-card">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">上传文档</div>
                <div class="panel-subtitle">支持 TXT、DOCX、PDF，识别后可直接导出脱敏文件。</div>
              </div>
            </div>
          </template>

          <el-upload
            class="upload-area"
            drag
            :auto-upload="false"
            :limit="1"
            accept=".pdf,.docx,.txt"
            :on-change="handleFileChange"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">
              将文件拖到这里，或 <em>点击选择文件</em>
            </div>
            <template #tip>
              <div class="el-upload__tip">单个文件最大 50MB。</div>
            </template>
          </el-upload>

          <el-alert
            v-if="templateName || hasOperatorConfig"
            class="mb-16"
            type="info"
            :closable="false"
            :title="templateName ? `当前模板：${templateName}` : '已启用自定义脱敏配置'"
          />

          <el-form label-width="120px">
            <el-form-item label="启用大模型">
              <el-switch v-model="options.use_llm" />
            </el-form-item>

            <el-form-item label="识别模型">
              <el-select
                v-model="selectedLlmModel"
                class="full-width"
                placeholder="请选择模型"
                :disabled="!options.use_llm || !llmModelOptions.length"
              >
                <el-option
                  v-for="model in llmModelOptions"
                  :key="model.name"
                  :label="formatModelLabel(model)"
                  :value="model.name"
                  :disabled="modelCatalog?.backend === 'ollama' && !model.installed"
                />
              </el-select>
              <div class="form-tip">
                {{
                  selectedModelDescription ||
                  '识别与上下文脱敏会统一使用这里选择的模型。4B 默认走高精度多轮策略，处理时间会更长，但更偏向减少漏检。'
                }}
              </div>
            </el-form-item>

            <el-form-item label="启用自定义规则">
              <el-switch v-model="options.use_custom" />
            </el-form-item>

            <el-form-item label="匿名策略">
              <el-radio-group v-model="selectedAnonymizationStrategy">
                <el-radio-button
                  v-for="option in anonymizationStrategyOptions"
                  :key="option.value"
                  :label="option.value"
                >
                  {{ option.label }}
                </el-radio-button>
              </el-radio-group>
              <div class="form-tip">
                {{
                  selectedAnonymizationDescription ||
                  '控制人物和主体名称的替换表达方式。'
                }}
              </div>
            </el-form-item>

            <el-form-item label="当前文件">
              <span>{{ currentFile?.name || '未选择文件' }}</span>
            </el-form-item>
          </el-form>

          <el-alert
            v-if="options.use_llm && modelCatalog && !modelCatalog.service_available"
            class="mb-16"
            type="warning"
            :closable="false"
            title="当前未连接到 Ollama 服务，模型安装状态可能无法实时获取。"
          />

          <el-alert
            v-if="runtimeStatus && !runtimeStatus.ready"
            class="mb-16"
            type="warning"
            :closable="false"
            :title="runtimeStatus.recommended_action"
          />

          <div class="action-stack">
            <el-button
              type="primary"
              size="large"
              :loading="analyzing"
              :disabled="!currentFile || !runtimeReady"
              @click="uploadFile"
            >
              开始识别
            </el-button>

            <el-button
              size="large"
              :loading="processing"
              :disabled="!analysisResult || !runtimeReady"
              @click="processCurrentTask"
            >
              生成脱敏文件
            </el-button>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="16">
        <el-card class="panel-card">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">识别结果</div>
                <div class="panel-subtitle">识别完成后，这里会展示实体、统计信息和文本预览。</div>
              </div>
              <el-tag v-if="analysisResult" type="success" size="large">
                {{ analysisResult.entities.length }} 个实体
              </el-tag>
            </div>
          </template>

          <el-empty v-if="!analysisResult" description="请先上传文件并开始识别。" />

          <div v-else>
            <el-descriptions :column="3" border class="mb-16">
              <el-descriptions-item label="文件名">
                {{ analysisResult.filename }}
              </el-descriptions-item>
              <el-descriptions-item label="文件类型">
                {{ fileTypeLabel }}
              </el-descriptions-item>
              <el-descriptions-item label="文本长度">
                {{ analysisResult.text.length }} 字符
              </el-descriptions-item>
            </el-descriptions>

            <el-alert
              v-if="analysisResult.llm_model"
              class="mb-16"
              type="info"
              :closable="false"
              :title="`本次识别使用模型：${analysisResult.llm_model}${analysisResult.llm_strategy_label ? `，策略：${analysisResult.llm_strategy_label}` : ''}`"
            />

            <el-alert
              v-if="analysisResult.anonymization_strategy_label"
              class="mb-16"
              type="info"
              :closable="false"
              :title="`本次匿名策略：${analysisResult.anonymization_strategy_label}`"
            />

            <el-alert
              v-if="documentTypeSummary"
              class="mb-16"
              type="info"
              :closable="false"
              :title="documentTypeSummary"
            />

            <el-alert
              v-if="analysisQualitySummary"
              class="mb-16"
              type="info"
              :closable="false"
              :title="analysisQualitySummary"
            />

            <el-alert
              v-if="analysisResult.entities.length === 0"
              type="warning"
              :closable="false"
              class="mb-16"
              title="识别完成，但当前没有命中任何实体。你仍然可以继续生成文件。"
            />

            <el-tabs v-model="activeTab">
              <el-tab-pane label="实体列表" name="entities">
                <el-table
                  :data="analysisResult.entities"
                  max-height="420"
                  empty-text="暂无识别结果"
                >
                  <el-table-column label="类型" width="160">
                    <template #default="{ row }">
                      <el-tag :type="getTagType(row.type)">{{ getTypeName(row.type) }}</el-tag>
                    </template>
                  </el-table-column>
                  <el-table-column prop="text" label="内容" min-width="220" />
                  <el-table-column label="替换预览" min-width="220">
                    <template #default="{ row }">
                      <span v-if="row.replacement">{{ row.replacement }}</span>
                      <span v-else-if="row.replacement_method === 'preserve'" class="secondary-text">
                        保留原文
                      </span>
                      <span v-else class="secondary-text">生成脱敏结果后显示</span>
                    </template>
                  </el-table-column>
                  <el-table-column label="来源" width="120">
                    <template #default="{ row }">
                      <el-tag size="small" :type="getSourceType(row.source)">
                        {{ getSourceName(row.source) }}
                      </el-tag>
                    </template>
                  </el-table-column>
                  <el-table-column label="置信度" width="110">
                    <template #default="{ row }">
                      {{ formatScore(row.score) }}
                    </template>
                  </el-table-column>
                </el-table>
              </el-tab-pane>

              <el-tab-pane label="文本预览" name="preview">
                <div class="text-preview">
                  <pre>{{ highlightedText }}</pre>
                </div>
              </el-tab-pane>

              <el-tab-pane label="统计信息" name="statistics">
                <el-descriptions :column="2" border>
                  <el-descriptions-item
                    v-for="item in statisticsRows"
                    :key="item.key"
                    :label="item.key"
                  >
                    {{ item.value }}
                  </el-descriptions-item>
                </el-descriptions>
              </el-tab-pane>
            </el-tabs>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-dialog
      v-model="resultDialogVisible"
      title="脱敏完成"
      width="880px"
      destroy-on-close
    >
      <template v-if="desensitizeResult">
        <el-alert
          title="脱敏文件已生成。"
          type="success"
          :closable="false"
          class="mb-16"
        />

        <el-alert
          v-if="desensitizeResult.warning"
          :title="desensitizeResult.warning"
          type="warning"
          :closable="false"
          class="mb-16"
        />

        <el-descriptions :column="3" border class="mb-16">
          <el-descriptions-item label="导出文件">
            {{ desensitizeResult.output_filename || '未生成' }}
          </el-descriptions-item>
          <el-descriptions-item label="导出格式">
            {{ (desensitizeResult.output_file_type || 'txt').toUpperCase() }}
          </el-descriptions-item>
          <el-descriptions-item label="保留原格式">
            {{ desensitizeResult.preserves_format ? '是' : '否' }}
          </el-descriptions-item>
        </el-descriptions>

        <el-alert
          :title="processingSummary"
          type="info"
          :closable="false"
          class="mb-16"
        />

        <div class="result-section">
          <div class="section-title">替换对照</div>
          <el-table
            :data="replacementRows"
            max-height="280"
            empty-text="当前没有可展示的替换对照"
          >
            <el-table-column label="类型" width="120">
              <template #default="{ row }">
                <el-tag :type="getTagType(row.type)">{{ getTypeName(row.type) }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="text" label="原文" min-width="220" />
            <el-table-column prop="replacement" label="替换后" min-width="240" />
            <el-table-column label="上下文" min-width="180">
              <template #default="{ row }">
                <div>{{ row.context_label || '未命名字段' }}</div>
                <div class="secondary-text">{{ row.context_role || '通用上下文' }}</div>
              </template>
            </el-table-column>
          </el-table>
        </div>

        <div class="result-section">
          <div class="section-title">文本预览</div>
          <div class="text-preview">
            <pre>{{ desensitizeResult.anonymized_text || '' }}</pre>
          </div>
        </div>
      </template>

      <template #footer>
        <el-button @click="resultDialogVisible = false">关闭</el-button>
        <el-button type="primary" @click="openDownload">下载结果</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import type { UploadFile } from 'element-plus'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import {
  downloadResult as buildDownloadUrl,
  getAvailableModels,
  getRuntimeStatus,
  processDesensitize as processDesensitizeApi,
  uploadAndAnalyze
} from '@/api/desensitize'
import type {
  AnalyzeResponse,
  DesensitizeResponse,
  Entity,
  LLMModelListResponse,
  LLMModelOption,
  RuntimeStatusResponse
} from '@/api/desensitize'
import { loadAppSettings } from '@/utils/settings'

const currentFile = ref<File | null>(null)
const analyzing = ref(false)
const processing = ref(false)
const analysisResult = ref<AnalyzeResponse | null>(null)
const desensitizeResult = ref<DesensitizeResponse | null>(null)
const resultDialogVisible = ref(false)
const activeTab = ref('entities')
const operatorConfig = ref<Record<string, any>>({})
const templateName = ref<string | null>(null)
const modelCatalog = ref<LLMModelListResponse | null>(null)
const runtimeStatus = ref<RuntimeStatusResponse | null>(null)
const selectedLlmModel = ref('')
const selectedAnonymizationStrategy = ref('official')

const anonymizationStrategyOptions = [
  {
    value: 'official',
    label: '正式文书风格',
    description:
      '主体名称尽量保留原有阅读感，只对关键部分做“某化”处理，适合正式材料。'
  },
  {
    value: 'serial_roles',
    label: '甲乙丙主体策略',
    description:
      '人物和主体优先改成甲乙丙类称谓，适合快速区分不同参与方。'
  }
]

const options = ref({
  use_llm: true,
  use_custom: true
})

const hasOperatorConfig = computed(() => Object.keys(operatorConfig.value).length > 0)
const llmModelOptions = computed(() => modelCatalog.value?.models || [])
const runtimeReady = computed(() => runtimeStatus.value?.ready ?? false)
const selectedModelOption = computed(
  () => llmModelOptions.value.find((item) => item.name === selectedLlmModel.value) || null
)

const selectedModelDescription = computed(() => {
  if (!selectedModelOption.value) {
    return ''
  }
  return `${selectedModelOption.value.strategy_label}：${selectedModelOption.value.strategy_description}`
})

const selectedAnonymizationDescription = computed(() => {
  return (
    anonymizationStrategyOptions.find((item) => item.value === selectedAnonymizationStrategy.value)
      ?.description || ''
  )
})

const fileTypeLabel = computed(() => {
  const fileType = analysisResult.value?.metadata?.file_type
  if (!fileType) {
    return '-'
  }
  return String(fileType).toUpperCase()
})

const documentTypeSummary = computed(() => {
  const metadata = analysisResult.value?.metadata || {}
  if (!metadata.llm_document_type_label) {
    return ''
  }

  const confidence = metadata.llm_document_type_confidence
    ? `，置信度 ${metadata.llm_document_type_confidence}`
    : ''
  const reason = metadata.llm_document_type_reason
    ? `，依据：${metadata.llm_document_type_reason}`
    : ''

  return `文档类型研判：${metadata.llm_document_type_label}${confidence}${reason}`
})

const analysisQualitySummary = computed(() => {
  const metadata = analysisResult.value?.metadata || {}
  const parts: string[] = []
  const recallPasses = Number(metadata.recall_passes ?? metadata.llm_recall_passes ?? 0)
  const specializedPasses = Array.isArray(metadata.llm_specialized_passes)
    ? metadata.llm_specialized_passes.length
    : 0
  const highRiskBlocks = Array.isArray(metadata.high_risk_blocks)
    ? metadata.high_risk_blocks.length
    : 0
  const definitionHints = Array.isArray(metadata.definition_hints)
    ? metadata.definition_hints.length
    : 0

  if (metadata.engine_strategy === 'precision_4b') {
    parts.push('当前使用 4B 高精度多轮策略')
  }

  if (recallPasses > 0) {
    parts.push(`召回轮次 ${recallPasses} 次`)
  }
  if (specializedPasses > 0) {
    parts.push(`专项轮次 ${specializedPasses} 组`)
  }
  if (highRiskBlocks > 0) {
    parts.push(`高风险片段 ${highRiskBlocks} 处`)
  }
  if (definitionHints > 0) {
    parts.push(`简称/定义锚点 ${definitionHints} 处`)
  }

  return parts.join('，')
})

const statisticsRows = computed(() => {
  if (!analysisResult.value) {
    return []
  }

  return Object.entries(analysisResult.value.statistics || {}).map(([key, value]) => {
    const count =
      value && typeof value === 'object' && 'count' in value
        ? (value as { count: number }).count
        : value

    return {
      key: getTypeName(key),
      value: typeof count === 'number' ? `${count}` : String(count)
    }
  })
})

const replacementRows = computed(() => {
  const entities = desensitizeResult.value?.entities || []
  if (entities.length === 0) {
    return []
  }

  const uniqueRows = new Map<string, Entity>()
  ;[...entities]
    .sort((a, b) => a.start - b.start)
    .forEach((entity) => {
      if (!entity.replacement || entity.replacement === entity.text) {
        return
      }

      const key = [entity.type, entity.text, entity.replacement].join('::')
      if (!uniqueRows.has(key)) {
        uniqueRows.set(key, entity)
      }
    })

  return [...uniqueRows.values()]
})

const processingSummary = computed(() => {
  if (!desensitizeResult.value) {
    return ''
  }

  const metadata = desensitizeResult.value.metadata || {}
  const qualityParts: string[] = []
  const residualHits = Array.isArray(metadata.residual_hits) ? metadata.residual_hits.length : 0
  const consistencyIssues = Array.isArray(metadata.consistency_issues)
    ? metadata.consistency_issues.length
    : 0
  const repairRounds = Array.isArray(metadata.repair_rounds) ? metadata.repair_rounds.length : 0
  const qualityGatePassed =
    typeof metadata.quality_gate_passed === 'boolean' ? metadata.quality_gate_passed : true
  const qualityGateReason = String(metadata.quality_gate_reason || '').trim()
  const recallPasses = Number(metadata.recall_passes ?? metadata.llm_recall_passes ?? 0)

  if (metadata.engine_strategy === 'precision_4b') {
    qualityParts.push('已启用 4B 高精度多轮闭环')
  }
  if (recallPasses > 0) {
    qualityParts.push(`召回轮次 ${recallPasses} 次`)
  }
  if (repairRounds > 0) {
    qualityParts.push(`自动修复 ${repairRounds} 轮`)
  }
  if (residualHits === 0) {
    qualityParts.push('残留复扫未发现未覆盖命中')
  } else {
    qualityParts.push(`残留复扫仍发现 ${residualHits} 处待关注命中`)
  }
  if (consistencyIssues === 0) {
    qualityParts.push('同一主体替换一致性检查通过')
  } else {
    qualityParts.push(`一致性检查提示 ${consistencyIssues} 项问题`)
  }
  if (!qualityGatePassed) {
    qualityParts.push(
      qualityGateReason ? `质量闸未完全通过：${qualityGateReason}` : '质量闸未完全通过'
    )
  }

  const qualityText = qualityParts.length ? `${qualityParts.join('，')}。` : ''
  const caseNumberHint = '案号和合同编号仅对数字部分脱敏，文字结构保留。'

  if (desensitizeResult.value.llm_assisted) {
    const modelText = desensitizeResult.value.llm_model
      ? `，模型为 ${desensitizeResult.value.llm_model}`
      : ''
    const strategyText = desensitizeResult.value.llm_strategy_label
      ? `，策略为 ${desensitizeResult.value.llm_strategy_label}`
      : ''
    const anonymizationText = desensitizeResult.value.anonymization_strategy_label
      ? `，匿名策略为 ${desensitizeResult.value.anonymization_strategy_label}`
      : ''

    return `本次处理已启用大模型参与识别与上下文脱敏${modelText}${strategyText}${anonymizationText}。金额、日期等判断性数值默认保留原文，${caseNumberHint}${qualityText}`
  }

  const anonymizationText = desensitizeResult.value.anonymization_strategy_label
    ? `当前匿名策略为 ${desensitizeResult.value.anonymization_strategy_label}。`
    : ''

  return `${anonymizationText}本次处理未启用大模型，当前替换结果主要来自上下文规则与结构化脱敏逻辑。金额和日期默认保留原文，${caseNumberHint}${qualityText}`
})

const highlightedText = computed(() => {
  if (!analysisResult.value) {
    return ''
  }

  const entities = [...analysisResult.value.entities].sort((a, b) => a.start - b.start)
  let cursor = 0
  let output = ''

  entities.forEach((entity) => {
    if (entity.start < cursor) {
      return
    }

    output += analysisResult.value?.text.slice(cursor, entity.start) || ''
    output += `[[${getTypeName(entity.type)}: ${entity.text}]]`
    cursor = entity.end
  })

  output += analysisResult.value.text.slice(cursor)
  return output
})

const formatScore = (value: number) => `${Math.round(value * 100)}%`

const handleFileChange = (file: UploadFile) => {
  currentFile.value = file.raw instanceof File ? file.raw : null
  analysisResult.value = null
  desensitizeResult.value = null
  resultDialogVisible.value = false
  activeTab.value = 'entities'
}

const loadDefaultSettings = () => {
  const settings = loadAppSettings()
  options.value.use_llm = settings.use_llm_default
  options.value.use_custom = settings.use_custom_default
  selectedLlmModel.value = settings.llm_model_default
  selectedAnonymizationStrategy.value = settings.anonymization_strategy_default
  operatorConfig.value = settings.operator_config || {}
  templateName.value = settings.template_name
}

const syncSelectedModel = (preferredModel?: string | null) => {
  const installedOptions = llmModelOptions.value.filter((item) => item.installed)
  const allOptionNames = llmModelOptions.value.map((item) => item.name)
  const installedOptionNames = installedOptions.map((item) => item.name)

  if (preferredModel && installedOptionNames.includes(preferredModel)) {
    selectedLlmModel.value = preferredModel
    return
  }

  const defaultModel = modelCatalog.value?.default_model
  if (defaultModel && installedOptionNames.includes(defaultModel)) {
    selectedLlmModel.value = defaultModel
    return
  }

  if (installedOptionNames.length > 0) {
    selectedLlmModel.value = installedOptionNames[0]
    return
  }

  selectedLlmModel.value = allOptionNames[0] || ''
}

const loadModelCatalog = async () => {
  try {
    modelCatalog.value = await getAvailableModels()
    syncSelectedModel(selectedLlmModel.value)
  } catch (error) {
    ElMessage.error('加载模型列表失败')
  }
}

const loadRuntimeStatus = async () => {
  try {
    runtimeStatus.value = await getRuntimeStatus()
  } catch (error) {
    ElMessage.error('读取运行环境状态失败')
  }
}

const formatModelLabel = (model: LLMModelOption) => {
  const status = model.installed ? '已安装' : '未安装'
  const defaultTag = model.is_default ? ' / 默认' : ''
  return `${model.name} (${model.strategy_label} / ${status}${defaultTag})`
}

const uploadFile = async () => {
  if (!currentFile.value) {
    ElMessage.warning('请先选择文件')
    return
  }

  if (!runtimeReady.value) {
    ElMessage.warning('当前运行环境尚未就绪，请先完成启动检查。')
    return
  }

  if (options.value.use_llm && !selectedLlmModel.value) {
    ElMessage.warning('请先选择识别模型')
    return
  }

  const currentModel = llmModelOptions.value.find((item) => item.name === selectedLlmModel.value)
  if (
    options.value.use_llm &&
    modelCatalog.value?.backend === 'ollama' &&
    currentModel &&
    !currentModel.installed
  ) {
    ElMessage.warning(
      `模型 ${selectedLlmModel.value} 尚未下载，请先执行：ollama pull ${selectedLlmModel.value}`
    )
    return
  }

  analyzing.value = true
  try {
    analysisResult.value = await uploadAndAnalyze(
      currentFile.value,
      options.value.use_llm,
      options.value.use_custom,
      options.value.use_llm ? selectedLlmModel.value : undefined,
      selectedAnonymizationStrategy.value
    )
    if (analysisResult.value.llm_model) {
      selectedLlmModel.value = analysisResult.value.llm_model
    }
    if (analysisResult.value.anonymization_strategy) {
      selectedAnonymizationStrategy.value = analysisResult.value.anonymization_strategy
    }
    activeTab.value = 'entities'
    ElMessage.success('识别完成')
  } finally {
    analyzing.value = false
  }
}

const processCurrentTask = async () => {
  if (!analysisResult.value) {
    return
  }

  processing.value = true
  try {
    desensitizeResult.value = await processDesensitizeApi({
      task_id: analysisResult.value.task_id,
      entities: analysisResult.value.entities,
      config: hasOperatorConfig.value ? operatorConfig.value : undefined,
      llm_model: analysisResult.value.llm_model || selectedLlmModel.value || undefined,
      anonymization_strategy: selectedAnonymizationStrategy.value
    })
    if (desensitizeResult.value.entities?.length) {
      analysisResult.value.entities = desensitizeResult.value.entities
    }
    if (desensitizeResult.value.llm_model) {
      selectedLlmModel.value = desensitizeResult.value.llm_model
    }
    if (desensitizeResult.value.anonymization_strategy) {
      selectedAnonymizationStrategy.value = desensitizeResult.value.anonymization_strategy
    }
    resultDialogVisible.value = true
    ElMessage.success('脱敏完成')
  } finally {
    processing.value = false
  }
}

const openDownload = () => {
  if (!analysisResult.value) {
    return
  }

  const url = desensitizeResult.value?.download_url || buildDownloadUrl(analysisResult.value.task_id)
  window.open(url, '_blank')
}

const getTypeName = (type: string) => {
  const typeMap: Record<string, string> = {
    PERSON: '人名',
    PERSON_NAME: '人名',
    ORGANIZATION: '组织机构',
    COMPANY_NAME: '公司名称',
    LOCATION: '地址',
    POSITION: '职位',
    CN_ID_CARD: '身份证号',
    CN_PHONE: '手机号',
    LANDLINE_PHONE: '座机号',
    CN_BANK_CARD: '银行卡号',
    CN_CREDIT_CODE: '统一社会信用代码',
    EMAIL_ADDRESS: '邮箱',
    AMOUNT: '金额',
    PROJECT_CODE: '项目代号',
    CONTRACT_NO: '合同编号',
    PRODUCT_NAME: '产品名称',
    SENSITIVE_TERM: '敏感术语',
    PROJECT: '项目名称',
    BANK_NAME: '开户行',
    ACCOUNT_NAME: '户名'
  }

  return typeMap[type] || type
}

const getTagType = (type: string) => {
  const tagMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info' | 'primary'> = {
    PERSON: 'success',
    PERSON_NAME: 'success',
    ORGANIZATION: 'warning',
    COMPANY_NAME: 'warning',
    LOCATION: 'info',
    POSITION: 'primary',
    CN_ID_CARD: 'danger',
    CN_PHONE: 'danger',
    LANDLINE_PHONE: 'danger',
    CN_BANK_CARD: 'danger',
    CONTRACT_NO: 'primary',
    PROJECT: 'info',
    BANK_NAME: 'warning',
    ACCOUNT_NAME: 'warning'
  }

  return tagMap[type] || ''
}

const getSourceName = (source: string) => {
  const sourceMap: Record<string, string> = {
    regex: '规则',
    custom: '自定义',
    contract: '合同字段',
    propagate: '重复补全',
    llm_review: '大模型复审',
    llm: '大模型',
    ollama: '大模型',
    ollama_definition: '定义锚点',
    memory_residual: '残留回扫',
    residual_repair: '质量修复'
  }

  return sourceMap[source] || source
}

const getSourceType = (source: string) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'primary' | 'info'> = {
    regex: 'success',
    custom: 'warning',
    contract: 'info',
    propagate: 'warning',
    llm_review: 'primary',
    llm: 'primary',
    ollama: 'primary',
    ollama_definition: 'primary',
    memory_residual: 'warning',
    residual_repair: 'warning'
  }

  return typeMap[source] || 'info'
}

onMounted(async () => {
  loadDefaultSettings()
  await loadRuntimeStatus()
  await loadModelCatalog()
})
</script>

<style scoped>
.desensitize-page {
  max-width: 1500px;
}

.panel-card {
  min-height: 680px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.panel-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
}

.panel-subtitle {
  margin-top: 4px;
  font-size: 13px;
  color: #6b7280;
}

.upload-area {
  margin-bottom: 20px;
}

.full-width {
  width: 100%;
}

.form-tip {
  margin-top: 8px;
  font-size: 12px;
  line-height: 1.6;
  color: #6b7280;
}

.action-stack {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.action-stack .el-button {
  width: 100%;
}

.text-preview {
  max-height: 420px;
  overflow-y: auto;
  border-radius: 8px;
  background: #f5f7fa;
  padding: 16px;
}

.text-preview pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.7;
  font-family: 'Consolas', 'Courier New', monospace;
}

.result-section {
  margin-bottom: 16px;
}

.section-title {
  margin-bottom: 12px;
  font-size: 14px;
  font-weight: 600;
  color: #1f2937;
}

.secondary-text {
  color: #909399;
  font-size: 12px;
}

.mb-16 {
  margin-bottom: 16px;
}

@media (max-width: 992px) {
  .panel-card {
    min-height: auto;
  }

  .panel-header {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
