<template>
  <div class="assistant-page">
    <el-row :gutter="20">
      <el-col :xs="24" :lg="8">
        <el-card class="panel-card">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">律师辅助</div>
                <div class="panel-subtitle">
                  诉讼/执行优先的独立工作流，单次分析会依次完成文书接入、分类、要素抽取、请求拆解和程序核对。
                </div>
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
            <div class="el-upload__text">将文件拖到这里，或 <em>点击选择文件</em></div>
            <template #tip>
              <div class="el-upload__tip">单个文件最大 50MB。首期优先支持诉讼、执行、保全和证据材料。</div>
            </template>
          </el-upload>

          <el-form label-width="110px">
            <el-form-item :label="isHighQualityLowmem ? '精审模型（按需）' : '协助模型'">
              <el-select
                v-model="selectedAssistantModel"
                class="full-width"
                :placeholder="isHighQualityLowmem ? '请选择按需精审模型' : '请选择 27B 模型'"
                :disabled="!assistantModelOptions.length"
              >
                <el-option
                  v-for="model in assistantModelOptions"
                  :key="model.name"
                  :label="formatModelLabel(model)"
                  :value="model.name"
                  :disabled="!model.installed"
                />
              </el-select>
              <div class="form-tip">
                {{
                  assistantModelDescription ||
                  (isHighQualityLowmem
                    ? '当前模式使用本地小模型做局部精审，不复用 Ollama 4B 全文抽取。'
                    : '律师辅助分区仅支持已安装的 27B 模型，且不复用文本脱敏的分析任务。')
                }}
              </div>
            </el-form-item>

            <el-form-item label="当前文件">
              <span>{{ currentFile?.name || assistantResult?.filename || '未选择文件' }}</span>
            </el-form-item>
          </el-form>

          <el-alert
            v-if="runtimeStatus && !runtimeStatus.ready"
            class="mb-16"
            type="warning"
            :closable="false"
            :title="runtimeStatus.recommended_action"
          />

          <el-alert
            v-else-if="!selectedAssistantModel"
            class="mb-16"
            type="warning"
            :closable="false"
            :title="
              isHighQualityLowmem
                ? '当前未检测到已安装的按需精审模型，请先完成高质量低内存模型下载。'
                : '当前未检测到已安装的 27B 协助模型，请先在 Ollama 中安装 qwen3.5:27b 系列模型。'
            "
          />

          <div class="action-stack">
            <el-button
              type="primary"
              size="large"
              :loading="assistantPending"
              :disabled="!currentFile || !runtimeReady || !selectedAssistantModel"
              @click="startAssistant"
            >
              开始律师辅助分析
            </el-button>
          </div>

          <el-divider content-position="left">阶段进度</el-divider>
          <el-steps direction="vertical" :active="currentStageIndex" finish-status="success">
            <el-step
              v-for="(stage, index) in stageDefinitions"
              :key="stage.key"
              :title="stage.label"
              :description="currentStageKey === stage.key ? currentStageMessage : stage.description"
              :status="assistantResult && index <= currentStageIndex ? 'success' : undefined"
            />
          </el-steps>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="16">
        <el-card class="panel-card">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">协助结果</div>
                <div class="panel-subtitle">
                  输出案件首页、请求事项拆解、程序信息核对、证据缺口和待补材料清单。
                </div>
              </div>
              <el-tag v-if="assistantResult" type="success" size="large">
                {{ assistantResult.sections.length }} 个结果区块
              </el-tag>
            </div>
          </template>

          <div v-if="assistantTaskStatus && assistantPending" class="mb-16">
            <el-alert
              type="info"
              :closable="false"
              :title="assistantTaskStatus.message || '正在执行律师辅助分析...'"
            />
            <el-progress class="mt-12" :percentage="assistantTaskStatus.progress" />
          </div>

          <el-empty
            v-if="!assistantResult && !assistantTaskStatus"
            description="请先上传文件并开始律师辅助分析。"
          />

          <template v-else-if="assistantResult">
            <el-alert
              class="mb-16"
              :type="assistantResult.support_mode === 'supported' ? 'success' : 'warning'"
              :closable="false"
              :title="assistantResult.support_notice"
            />

            <el-alert
              class="mb-16"
              type="info"
              :closable="false"
              :title="assistantResult.summary"
            />

            <el-descriptions :column="3" border class="mb-16">
              <el-descriptions-item label="文件名">
                {{ assistantResult.filename }}
              </el-descriptions-item>
              <el-descriptions-item label="文书类型">
                {{ assistantResult.document_type_label }}
              </el-descriptions-item>
              <el-descriptions-item label="支持模式">
                <el-tag :type="assistantResult.support_mode === 'supported' ? 'success' : 'warning'">
                  {{ assistantResult.support_mode === 'supported' ? '重点支持' : '材料概览' }}
                </el-tag>
              </el-descriptions-item>
              <el-descriptions-item label="协助模型">
                {{ assistantResult.metadata.assistant_model || '-' }}
              </el-descriptions-item>
              <el-descriptions-item label="OCR 模式">
                {{ assistantOcrModeLabel }}
              </el-descriptions-item>
              <el-descriptions-item label="证据片段数">
                {{ assistantResult.metadata.evidence_count ?? 0 }}
              </el-descriptions-item>
            </el-descriptions>

            <el-alert
              v-if="assistantResult.metadata.limited_reason"
              class="mb-16"
              type="warning"
              :closable="false"
              :title="assistantResult.metadata.limited_reason"
            />

            <div class="section-stack">
              <el-card
                v-for="section in assistantResult.sections"
                :key="section.type"
                shadow="never"
                class="section-card"
              >
                <template #header>
                  <div class="section-header">
                    <div>{{ section.title }}</div>
                    <el-tag size="small">{{ section.type }}</el-tag>
                  </div>
                </template>

                <div class="item-stack">
                  <div
                    v-for="(item, index) in section.items"
                    :key="`${section.type}-${index}`"
                    class="result-item"
                  >
                    <div class="item-title-row">
                      <div class="item-title">
                        {{ item.title || item.label || '项目' }}
                      </div>
                      <div class="tag-list">
                        <el-tag
                          v-if="item.status"
                          size="small"
                          :type="getStatusTagType(item.status)"
                        >
                          {{ formatStatusLabel(item.status) }}
                        </el-tag>
                        <el-tag
                          v-if="item.severity"
                          size="small"
                          :type="getSeverityType(item.severity)"
                        >
                          {{ item.severity }}
                        </el-tag>
                      </div>
                    </div>
                    <div v-if="item.value" class="item-value">{{ item.value }}</div>
                    <div v-if="item.reason" class="item-reason">{{ item.reason }}</div>
                    <div v-if="item.action_hint" class="item-hint">
                      建议操作：{{ item.action_hint }}
                    </div>
                    <div v-if="item.evidence_refs.length" class="evidence-actions">
                      <el-button
                        v-for="(evidence, evidenceIndex) in item.evidence_refs"
                        :key="`${section.type}-${index}-${evidenceIndex}`"
                        size="small"
                        type="primary"
                        plain
                        @click="openEvidenceDrawer(section.title, evidence)"
                      >
                        {{ formatEvidenceLabel(evidence, evidenceIndex) }}
                      </el-button>
                    </div>
                  </div>
                </div>
              </el-card>
            </div>

            <el-divider content-position="left">原文预览</el-divider>
            <div class="text-preview">
              <pre>{{ highlightedPreviewText }}</pre>
            </div>
          </template>
        </el-card>
      </el-col>
    </el-row>

    <el-drawer v-model="evidenceDrawerVisible" title="证据定位" size="520px">
      <template v-if="activeEvidence">
        <el-descriptions :column="1" border class="mb-16">
          <el-descriptions-item label="所属区块">
            {{ activeEvidenceSectionTitle || '-' }}
          </el-descriptions-item>
          <el-descriptions-item label="片段类型">
            {{ activeEvidence.block_type }}
          </el-descriptions-item>
          <el-descriptions-item label="页码">
            {{ activeEvidence.page ?? '未标页' }}
          </el-descriptions-item>
          <el-descriptions-item label="证据质量">
            {{ activeEvidence.evidence_quality }}
          </el-descriptions-item>
        </el-descriptions>
        <pre class="drawer-quote">{{ activeEvidence.quote }}</pre>
      </template>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage } from 'element-plus'
import type { UploadFile } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import {
  getAssistantResult,
  getAssistantStatus,
  type AssistantEvidenceRef,
  type AssistantResult,
  type AssistantTaskStatus,
  uploadForAssistant
} from '@/api/assistant'
import { useRuntimeModels } from '@/composables/useRuntimeModels'
import { useTaskPolling } from '@/composables/useTaskPolling'

const currentFile = ref<File | null>(null)
const assistantPending = ref(false)
const assistantTaskStatus = ref<AssistantTaskStatus | null>(null)
const assistantResult = ref<AssistantResult | null>(null)
const selectedAssistantModel = ref('')
const evidenceDrawerVisible = ref(false)
const activeEvidence = ref<AssistantEvidenceRef | null>(null)
const activeEvidenceSectionTitle = ref('')

const {
  runtimeStatus,
  isHighQualityLowmem,
  llmModelOptions,
  runtimeReady,
  syncSelectedModel,
  formatModelLabel,
  loadModelCatalog,
  loadRuntimeStatus
} = useRuntimeModels()

const assistantModelOptions = computed(() =>
  llmModelOptions.value.filter((item) => item.supports_precision_review)
)

const assistantModelDescription = computed(() => {
  const model = assistantModelOptions.value.find((item) => item.name === selectedAssistantModel.value)
  if (!model) {
    return ''
  }
  if (isHighQualityLowmem.value) {
    return `当前选择的是按需精审模型 ${model.name}，只用于局部片段复核，不承担脱敏主识别。`
  }
  return `${model.strategy_label}：${model.strategy_description}`
})

const stageDefinitions = [
  { key: 'intake', label: '文书接入与 OCR', description: '解析文件、恢复文本并准备 OCR。' },
  { key: 'classification', label: '文书分类与范围判断', description: '判断是否属于首期重点支持范围。' },
  { key: 'extract', label: '案件要素提取', description: '抽取主体、案号、时间、金额等要素。' },
  { key: 'request', label: '请求事项拆解', description: '整理诉请、申请事项或核心办案事项。' },
  { key: 'finalize', label: '程序核对与缺口整理', description: '输出核对项、证据缺口和待补材料。' }
]

const currentStageKey = computed(() => assistantTaskStatus.value?.stage_key || (assistantResult.value ? 'finalize' : 'intake'))
const currentStageMessage = computed(() => assistantTaskStatus.value?.message || '')
const currentStageIndex = computed(() => {
  const index = stageDefinitions.findIndex((item) => item.key === currentStageKey.value)
  if (index === -1) {
    return assistantResult.value ? stageDefinitions.length : 0
  }
  return assistantResult.value ? Math.min(index + 1, stageDefinitions.length) : index
})

const assistantOcrModeLabel = computed(() => {
  const mode = String(assistantResult.value?.metadata?.ocr_mode || '').toLowerCase()
  if (mode === 'ocr_pro') {
    return 'OCR Pro'
  }
  if (mode === 'ocr_standard') {
    return '标准 OCR'
  }
  if (mode === 'native_or_mixed') {
    return '原生文本 / 混合'
  }
  return '未标注'
})

const highlightedPreviewText = computed(() => {
  const text = assistantResult.value?.text || ''
  if (!text || !activeEvidence.value || activeEvidence.value.end <= activeEvidence.value.start) {
    return text
  }
  const start = Math.max(0, activeEvidence.value.start)
  const end = Math.max(start, activeEvidence.value.end)
  return `${text.slice(0, start)}[[证据片段]]${text.slice(start, end)}[[/证据片段]]${text.slice(end)}`
})

const assistantPoller = useTaskPolling<AssistantTaskStatus, AssistantResult>({
  fetchStatus: getAssistantStatus,
  fetchResult: getAssistantResult,
  getStatus: (status) => status.status,
  isReady: (status) => status.status === 'completed',
  onStatus: (status) => {
    assistantTaskStatus.value = status
  },
  onResult: (result) => {
    assistantResult.value = result
    assistantPending.value = false
    ElMessage.success('律师辅助分析完成')
  },
  onFailed: (status) => {
    assistantPending.value = false
    ElMessage.error(status.error_message || status.message || '律师辅助分析失败')
  },
  onError: () => {
    assistantPending.value = false
  }
})

const applyModelSelection = () => {
  selectedAssistantModel.value =
    syncSelectedModel(selectedAssistantModel.value, { requireInstalled: true, tier: 'review' }) ||
    selectedAssistantModel.value
}

const handleFileChange = (file: UploadFile) => {
  assistantPoller.stop()
  currentFile.value = file.raw instanceof File ? file.raw : null
  assistantPending.value = false
  assistantTaskStatus.value = null
  assistantResult.value = null
  activeEvidence.value = null
}

const startAssistant = async () => {
  if (!currentFile.value) {
    ElMessage.warning('请先选择文件')
    return
  }
  if (!runtimeReady.value) {
    ElMessage.warning('当前运行环境尚未就绪，请先完成启动检查。')
    return
  }
  if (!selectedAssistantModel.value) {
    ElMessage.warning(isHighQualityLowmem.value ? '请先选择已安装的按需精审模型。' : '请先选择已安装的 27B 协助模型。')
    return
  }

  assistantPending.value = true
  assistantResult.value = null

  try {
    const task = await uploadForAssistant(currentFile.value, selectedAssistantModel.value)
    assistantTaskStatus.value = task
    void assistantPoller.poll(task.assistant_id)
  } catch {
    assistantPending.value = false
  }
}

const openEvidenceDrawer = (sectionTitle: string, evidence: AssistantEvidenceRef) => {
  activeEvidenceSectionTitle.value = sectionTitle
  activeEvidence.value = evidence
  evidenceDrawerVisible.value = true
}

const formatEvidenceLabel = (evidence: AssistantEvidenceRef, index: number) => {
  const pageLabel = evidence.page ? `第 ${evidence.page} 页` : '未标页'
  return `${pageLabel} / 证据 ${index + 1}`
}

const getSeverityType = (severity?: string | null) => {
  const normalized = String(severity || '').toLowerCase()
  if (normalized === 'high') {
    return 'danger'
  }
  if (normalized === 'medium') {
    return 'warning'
  }
  return 'info'
}

const getStatusTagType = (status?: string | null) => {
  const normalized = String(status || '').toLowerCase()
  if (normalized === 'missing') {
    return 'danger'
  }
  if (normalized === 'needs_review' || normalized === 'warning') {
    return 'warning'
  }
  return 'success'
}

const formatStatusLabel = (status?: string | null) => {
  const normalized = String(status || '').toLowerCase()
  if (normalized === 'missing') {
    return '待补'
  }
  if (normalized === 'needs_review') {
    return '待核对'
  }
  if (normalized === 'warning') {
    return '提示'
  }
  return '已识别'
}

onMounted(async () => {
  await loadRuntimeStatus()
  await loadModelCatalog()
  applyModelSelection()
})
</script>

<style scoped>
.assistant-page {
  max-width: 1500px;
}

.panel-card {
  min-height: 760px;
}

.panel-header,
.section-header,
.item-title-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}

.panel-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
}

.panel-subtitle,
.form-tip,
.item-reason,
.item-hint {
  color: #6b7280;
  font-size: 13px;
  line-height: 1.6;
}

.upload-area,
.mb-16 {
  margin-bottom: 16px;
}

.mt-12 {
  margin-top: 12px;
}

.full-width {
  width: 100%;
}

.action-stack {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.action-stack .el-button {
  width: 100%;
}

.section-stack,
.item-stack {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.section-card {
  margin-bottom: 16px;
}

.result-item {
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 12px 14px;
  background: #f8fafc;
}

.item-title {
  font-weight: 600;
  color: #111827;
}

.item-value {
  margin-top: 6px;
  color: #1f2937;
  line-height: 1.7;
}

.tag-list,
.evidence-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.evidence-actions {
  margin-top: 12px;
}

.text-preview {
  max-height: 420px;
  overflow-y: auto;
  border-radius: 10px;
  background: #f5f7fa;
  padding: 16px;
}

.text-preview pre,
.drawer-quote {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
  line-height: 1.7;
  font-family: 'Consolas', 'Courier New', monospace;
}

@media (max-width: 992px) {
  .panel-card {
    min-height: auto;
  }
}
</style>
