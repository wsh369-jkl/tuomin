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

          <input
            ref="folderInputRef"
            class="hidden-folder-input"
            type="file"
            multiple
            webkitdirectory
            directory
            @change="handleFolderSelection"
          />

          <div class="folder-picker-card">
            <div class="folder-picker-title">文件夹批量处理</div>
            <div class="folder-picker-subtitle">
              选择某个文件夹后，系统会自动识别并脱敏其中所有 TXT、DOCX、PDF 文件。
            </div>
            <el-button class="full-width" plain @click="triggerFolderPicker">选择文件夹</el-button>
            <div class="form-tip">当前文件夹：{{ selectedFolderSummary }}</div>
          </div>

          <el-alert
            v-if="templateName || hasOperatorConfig"
            class="mb-16"
            type="info"
            :closable="false"
            :title="templateName ? `当前模板：${templateName}` : '已启用自定义脱敏配置'"
          />
	
		          <el-form label-width="120px">
		            <el-form-item label="启用高质量识别">
	              <el-switch v-model="options.use_llm" />
	            </el-form-item>

	            <el-form-item label="精审模型（按需）">
	              <el-select
	                v-model="selectedLlmModel"
	                class="full-width"
	                placeholder="请选择按需精审模型"
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
	                  '小 Qwen 只在需要时做片段补漏，是否实际参与会在识别结果里显示。'
	                }}
	              </div>
            </el-form-item>

            <el-form-item label="启用自定义规则">
              <el-switch v-model="options.use_custom" />
            </el-form-item>

            <el-form-item label="隐名线路">
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
                  '在这里切换正式文书线、甲乙丙主体线或编码隐名线。'
                }}
              </div>
            </el-form-item>

            <el-form-item label="当前文件">
              <span>{{ currentFile?.name || '未选择文件' }}</span>
            </el-form-item>

            <el-form-item label="当前文件夹">
              <span>{{ selectedFolderSummary }}</span>
            </el-form-item>
          </el-form>

          <el-alert
            v-if="modelServiceWarningTitle"
            class="mb-16"
            type="warning"
            :closable="false"
            :title="modelServiceWarningTitle"
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
	              :loading="analyzing || analysisPending"
	              :disabled="!currentFile || !runtimeReady"
	              @click="uploadFile"
	            >
              开始识别
            </el-button>

            <el-button
              size="large"
              :loading="processing"
              :disabled="!analysisResult || !runtimeReady || analysisPending"
              @click="processCurrentTask"
            >
              生成脱敏文件
            </el-button>

            <el-button
              size="large"
              type="primary"
              plain
              :loading="batchSubmitting || batchPending"
              :disabled="!currentFolderFiles.length || !runtimeReady"
              @click="startBatchProcessing"
            >
              批量识别并脱敏文件夹
            </el-button>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="16">
        <el-card class="panel-card">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">{{ batchPanelActive ? '批量处理结果' : '识别结果' }}</div>
                <div class="panel-subtitle">
                  {{
                    batchPanelActive
                      ? '文件夹内所有文件会自动完成识别、脱敏和导出，这里展示整体进度和每个文件的结果。'
                      : '识别完成后，这里会展示实体、统计信息和文本预览。'
                  }}
                </div>
              </div>
              <el-tag
                v-if="batchPanelActive && currentBatchView"
                :type="getBatchStatusTagType(currentBatchView.status)"
                size="large"
              >
                {{ formatTaskStatus(currentBatchView.status) }} · {{ currentBatchView.completed_count }}/{{ currentBatchView.file_count }}
              </el-tag>
              <el-tag v-else-if="analysisResult" type="success" size="large">
                {{ analysisResult.entities.length }} 个实体
              </el-tag>
            </div>
          </template>

          <div v-if="batchPanelActive">
            <template v-if="!currentBatchView">
              <el-empty description="已选择文件夹，点击左侧按钮开始批量识别并脱敏。" />
            </template>

            <template v-else>
              <el-alert
                class="mb-16"
                type="info"
                :closable="false"
                :title="currentBatchView.message || '后台正在批量处理，请稍候。'"
              />
              <div class="batch-progress-shell mb-16">
                <div class="batch-progress-header">
                  <div>
                    <div class="batch-progress-kicker">批量进度</div>
                    <div class="batch-progress-title">{{ batchProgressHeadline }}</div>
                    <div class="batch-progress-caption">{{ batchExecutionSummary }}</div>
                  </div>
                  <div class="batch-progress-ratio">
                    <span>{{ currentBatchView.completed_count }}</span>
                    <small>/ {{ currentBatchView.file_count }}</small>
                  </div>
                </div>

                <el-progress
                  class="batch-progress-bar"
                  :percentage="currentBatchView.progress"
                  :stroke-width="12"
                  :status="getBatchProgressStatus(currentBatchView.status)"
                />

                <div class="batch-stats-grid">
                  <div
                    v-for="metric in batchMetrics"
                    :key="metric.key"
                    class="batch-stat"
                    :class="`is-${metric.tone}`"
                  >
                    <div class="batch-stat-label">{{ metric.label }}</div>
                    <div class="batch-stat-value">{{ metric.value }}</div>
                  </div>
                </div>

                <div v-if="batchFocusItem" class="batch-focus-panel">
                  <div class="batch-focus-header">
                    <div>
                      <div class="batch-focus-kicker">{{ batchFocusLead }}</div>
                      <div class="batch-focus-path">{{ batchFocusItem.relative_path }}</div>
                    </div>
                    <el-tag size="small" :type="getBatchStatusTagType(batchFocusItem.status)">
                      {{ getBatchStageLabel(batchFocusItem.status) }}
                    </el-tag>
                  </div>

                  <div class="batch-focus-meta">
                    <span v-for="meta in batchFocusMeta" :key="meta">{{ meta }}</span>
                  </div>

                  <el-progress
                    :percentage="batchFocusItem.progress"
                    :stroke-width="8"
                    :status="getBatchProgressStatus(batchFocusItem.status)"
                  />

                  <div class="batch-step-track">
                    <div
                      v-for="step in batchFocusSteps"
                      :key="step.key"
                      class="batch-step-chip"
                      :class="{
                        'is-done': step.done,
                        'is-current': step.current,
                        'is-error': step.error
                      }"
                    >
                      <span class="batch-step-dot" />
                      <span>{{ step.label }}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div class="batch-toolbar">
                <el-button
                  v-if="batchResult?.archive_download_url"
                  type="primary"
                  @click="openBatchArchive"
                >
                  下载整包结果
                </el-button>
                <div class="secondary-text">当前状态：{{ formatTaskStatus(currentBatchView.status) }}</div>
              </div>

              <el-table
                :data="batchRows"
                max-height="520"
                empty-text="当前没有可展示的批量结果"
                :row-class-name="getBatchTableRowClass"
              >
                <el-table-column label="文件" min-width="280">
                  <template #default="{ row }">
                    <div class="batch-file-cell">
                      <div class="batch-file-name">{{ row.relative_path }}</div>
                      <div class="secondary-text">
                        {{
                          isBatchItemActive(row.status)
                            ? '当前正在处理'
                            : row.status === 'queued'
                              ? '等待轮到'
                              : row.status === 'failed'
                                ? '处理失败，建议复核'
                                : row.output_filename
                                  ? `已输出 ${row.output_filename}`
                                  : getBatchStageLabel(row.status)
                        }}
                      </div>
                    </div>
                  </template>
                </el-table-column>
                <el-table-column label="状态" width="120">
                  <template #default="{ row }">
                    <el-tag size="small" :type="getBatchStatusTagType(row.status)">
                      {{ formatTaskStatus(row.status) }}
                    </el-tag>
                  </template>
                </el-table-column>
                <el-table-column label="进度" width="180">
                  <template #default="{ row }">
                    <div class="batch-mini-progress">
                      <el-progress
                        :percentage="row.progress"
                        :stroke-width="6"
                        :show-text="false"
                        :status="getBatchProgressStatus(row.status)"
                      />
                      <div class="secondary-text">{{ row.progress }}% · {{ getBatchStageLabel(row.status) }}</div>
                    </div>
                  </template>
                </el-table-column>
                <el-table-column label="实体数" width="90">
                  <template #default="{ row }">
                    {{ row.entities_count || 0 }}
                  </template>
                </el-table-column>
                <el-table-column label="导出结果" min-width="210">
                  <template #default="{ row }">
                    <div>{{ row.output_filename || '-' }}</div>
                    <div class="secondary-text">
                      {{ row.output_file_type ? row.output_file_type.toUpperCase() : '' }}
                    </div>
                    <div v-if="row.mapping_output_filename" class="secondary-text">
                      对照目录：{{ row.mapping_output_filename }}
                    </div>
                  </template>
                </el-table-column>
                <el-table-column label="说明" min-width="220">
                  <template #default="{ row }">
                    <span>{{ row.warning || row.error_message || row.message || '-' }}</span>
                  </template>
                </el-table-column>
                <el-table-column label="操作" width="180">
                  <template #default="{ row }">
                    <div v-if="row.download_url || row.mapping_download_url">
                      <el-button
                        v-if="row.download_url"
                        link
                        type="primary"
                        @click="openBatchItem(row)"
                      >
                        下载结果
                      </el-button>
                      <el-button
                        v-if="row.mapping_download_url"
                        link
                        @click="openBatchItemMapping(row)"
                      >
                        对照目录
                      </el-button>
                    </div>
                    <span v-else class="secondary-text">-</span>
                  </template>
                </el-table-column>
              </el-table>
            </template>
          </div>

          <div v-else-if="!analysisResult && analysisTaskStatus">
            <el-alert
              class="mb-16"
              type="info"
              :closable="false"
              :title="analysisTaskStatus.message || '后台正在识别，请稍候。'"
            />
            <el-progress
              :percentage="analysisTaskStatus.progress"
              :status="analysisTaskStatus.status === 'failed' ? 'exception' : undefined"
            />
            <p class="secondary-text mt-12">
              当前状态：{{ formatTaskStatus(analysisTaskStatus.status) }}
            </p>
          </div>

          <el-empty v-else-if="!analysisResult" description="请先上传文件并开始识别。" />

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
              v-if="analysisReviewSummary"
              class="mb-16"
              :type="analysisReviewAlertType"
              :closable="false"
              :title="analysisReviewSummary"
            />

            <el-alert
              v-if="analysisOcrSummary"
              class="mb-16"
              :type="analysisOcrAlertType"
              :closable="false"
              :title="analysisOcrSummary"
            />

            <el-alert
              v-if="analysisWorkflowRecommendation"
              class="mb-16"
              type="warning"
              :closable="false"
              :title="analysisWorkflowRecommendation"
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
                <div class="statistics-section">
                  <div class="section-title">实体类型</div>
                  <el-descriptions :column="2" border>
                    <el-descriptions-item
                      v-for="item in statisticsRows"
                      :key="item.key"
                      :label="item.key"
                    >
                      {{ item.value }}
                    </el-descriptions-item>
                  </el-descriptions>
                </div>
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
          <el-descriptions-item label="脱敏文件">
            {{ desensitizeResult.output_filename || '未生成' }}
          </el-descriptions-item>
          <el-descriptions-item label="对照目录">
            {{ desensitizeResult.mapping_output_filename || '未生成' }}
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
        <el-button
          v-if="desensitizeResult?.mapping_download_url || desensitizeResult?.mapping_output_filename"
          @click="openMappingDownload"
        >
          下载对照目录
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRoute } from 'vue-router'
import type { UploadFile } from 'element-plus'
import { ElMessage } from 'element-plus'
import { UploadFilled } from '@element-plus/icons-vue'
import {
  downloadBatchArchive as buildBatchArchiveUrl,
  downloadBatchItem as buildBatchItemUrl,
  downloadBatchItemMapping as buildBatchItemMappingUrl,
  downloadMappingResult as buildMappingDownloadUrl,
  downloadResult as buildDownloadUrl,
  getAnalyzeResult,
  getBatchResult,
  getBatchTaskStatus,
  getProcessedResult,
  getTaskStatus,
  processDesensitizeAsync as processDesensitizeApi,
  uploadAndAnalyze,
  uploadAndProcessFolder
} from '@/api/desensitize'
import type {
  AnalyzeResponse,
  BatchFileItem,
  BatchResult,
  BatchTaskStatus,
  DesensitizeResponse,
  Entity,
  TaskStatus
} from '@/api/desensitize'
import { useRuntimeModels } from '@/composables/useRuntimeModels'
import { usePageSession } from '@/composables/usePageSession'
import { useTaskPolling } from '@/composables/useTaskPolling'
import { DEFAULT_DESENSITIZE_MODE, loadAppSettings, type DesensitizeMode } from '@/utils/settings'

const route = useRoute()
type SelectedFolderFile = {
  file: File
  relativePath: string
}

const currentFile = ref<File | null>(null)
const folderInputRef = ref<HTMLInputElement | null>(null)
const currentFolderName = ref('')
const currentFolderFiles = ref<SelectedFolderFile[]>([])
const analyzing = ref(false)
const analysisPending = ref(false)
const processing = ref(false)
const batchSubmitting = ref(false)
const batchPending = ref(false)
const analysisTaskStatus = ref<TaskStatus | null>(null)
const analysisResult = ref<AnalyzeResponse | null>(null)
const desensitizeResult = ref<DesensitizeResponse | null>(null)
const batchTaskStatus = ref<BatchTaskStatus | null>(null)
const batchResult = ref<BatchResult | null>(null)
const resultDialogVisible = ref(false)
const activeTab = ref('entities')
const operatorConfig = ref<Record<string, any>>({})
const templateName = ref<string | null>(null)
const selectedDesensitizeMode = ref<DesensitizeMode>(DEFAULT_DESENSITIZE_MODE)
const selectedLlmModel = ref('')
const selectedAnonymizationStrategy = ref('official')
const {
  modelCatalog,
  runtimeStatus,
  isHighQualityWorkflow,
  llmModelOptions,
  runtimeReady,
  syncSelectedModel: pickModelOption,
  formatModelLabel,
  loadModelCatalog: loadModelCatalogState,
  loadRuntimeStatus: loadRuntimeStatusState
} = useRuntimeModels()
const { pageSessionId } = usePageSession()

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
  },
  {
    value: 'symbolic_codes',
    label: '编码隐名线',
    description:
      '人名改成 a/b/c/d，机构改成 alpha/beta/gamma，地名改成甲地/乙地，并对同一主体保持稳定编码。'
  }
]

const options = ref({
  use_llm: true,
  use_custom: true
})

const BATCH_ACTIVE_STATUSES = new Set(['parsing', 'analyzing', 'anonymizing'])
const BATCH_TERMINAL_STATUSES = new Set(['completed', 'failed'])
const BATCH_STEP_SEQUENCE = [
  { key: 'queued', label: '排队' },
  { key: 'parsing', label: '解析文本' },
  { key: 'rules', label: '规则识别' },
  { key: 'primary', label: '主识别' },
  { key: 'review', label: '片段审查' },
  { key: 'quality', label: '质量复扫' },
  { key: 'anonymizing', label: '生成结果' },
  { key: 'completed', label: '完成' }
]

function isBatchItemActive(status: string) {
  return BATCH_ACTIVE_STATUSES.has(status)
}

function getBatchStageLabel(status: string) {
  const labels: Record<string, string> = {
    queued: '等待排队',
    parsing: '文本解析',
    analyzing: '主识别/片段审查',
    anonymizing: '生成文本结果',
    completed: '处理完成',
    failed: '处理失败'
  }

  return labels[status] || formatTaskStatus(status)
}

function getBatchProgressStatus(status: string): '' | 'success' | 'warning' | 'exception' | undefined {
  if (status === 'completed') {
    return 'success'
  }
  if (status === 'failed') {
    return 'exception'
  }
  if (isBatchItemActive(status) || status === 'processing') {
    return 'warning'
  }
  return undefined
}

function resolveBatchStepIndex(item: BatchFileItem | null) {
  if (!item) {
    return 0
  }

  if (item.status === 'completed') {
    return BATCH_STEP_SEQUENCE.length - 1
  }
  if (item.status === 'anonymizing') {
    return 6
  }
  if (item.status === 'analyzing') {
    if (item.progress >= 86) {
      return 5
    }
    if (item.progress >= 76) {
      return 4
    }
    if (item.progress >= 56) {
      return 3
    }
    return 2
  }
  if (item.status === 'parsing') {
    return 1
  }
  if (item.status === 'failed') {
    if (item.progress >= 96) {
      return 6
    }
    if (item.progress >= 86) {
      return 5
    }
    if (item.progress >= 76) {
      return 4
    }
    if (item.progress >= 72) {
      return 6
    }
    if (item.progress >= 56) {
      return 3
    }
    if (item.progress >= 38) {
      return 2
    }
    if (item.progress >= 10) {
      return 1
    }
  }

  return 0
}

const hasOperatorConfig = computed(() => Object.keys(operatorConfig.value).length > 0)
const selectedModelOption = computed(
  () => llmModelOptions.value.find((item) => item.name === selectedLlmModel.value) || null
)

const selectedModelDescription = computed(() => {
  if (!selectedModelOption.value) {
    return ''
  }
  return '按低内存三层工作流调度：小模型主召回，Qwen 只审风险片段。'
})

const modelServiceWarningTitle = computed(() => {
  if (!options.value.use_llm || !modelCatalog.value || modelCatalog.value.service_available) {
    return ''
  }
  if (isHighQualityWorkflow.value) {
    return '当前无法完整读取本地模型状态，主流程仍会优先使用已安装的中文实体模型；精审模型状态可能无法实时更新。'
  }
  return ''
})

const selectedAnonymizationDescription = computed(() => {
  return (
    anonymizationStrategyOptions.find((item) => item.value === selectedAnonymizationStrategy.value)
      ?.description || ''
  )
})

const batchPanelActive = computed(
  () => currentFolderFiles.value.length > 0 || !!batchTaskStatus.value || !!batchResult.value
)

const selectedFolderSummary = computed(() => {
  if (!currentFolderFiles.value.length) {
    return '未选择文件夹'
  }
  return `${currentFolderName.value || 'selected-folder'}（${currentFolderFiles.value.length} 个支持文件）`
})

const currentBatchView = computed(() => batchResult.value || batchTaskStatus.value)

const batchRows = computed(() => currentBatchView.value?.items || [])

const batchQueuedCount = computed(() => batchRows.value.filter((item) => item.status === 'queued').length)

const batchRunningCount = computed(() =>
  batchRows.value.filter((item) => isBatchItemActive(item.status)).length
)

const batchDoneCount = computed(() =>
  batchRows.value.filter((item) => item.status === 'completed').length
)

const batchFailedCount = computed(() =>
  batchRows.value.filter((item) => item.status === 'failed').length
)

const batchMetrics = computed(() => [
  { key: 'queued', label: '待处理', value: batchQueuedCount.value, tone: 'neutral' },
  { key: 'running', label: '处理中', value: batchRunningCount.value, tone: 'accent' },
  { key: 'done', label: '已完成', value: batchDoneCount.value, tone: 'success' },
  { key: 'failed', label: '失败', value: batchFailedCount.value, tone: 'danger' }
])

const batchActiveItem = computed(
  () => batchRows.value.find((item) => isBatchItemActive(item.status)) || null
)

const batchRecentTerminalItem = computed(
  () => [...batchRows.value].reverse().find((item) => BATCH_TERMINAL_STATUSES.has(item.status)) || null
)

const batchQueuedItem = computed(() => batchRows.value.find((item) => item.status === 'queued') || null)

const batchFocusItem = computed(
  () => batchActiveItem.value || batchRecentTerminalItem.value || batchQueuedItem.value || batchRows.value[0] || null
)

const batchFocusIndex = computed(() => {
  if (!batchFocusItem.value) {
    return -1
  }

  return batchRows.value.findIndex((item) => item.item_id === batchFocusItem.value?.item_id)
})

const batchProgressHeadline = computed(() => {
  const batch = currentBatchView.value
  if (!batch) {
    return ''
  }

  if (batch.status === 'completed') {
    return '全部文件已处理完成'
  }
  if (batch.status === 'failed') {
    return '批量处理失败，未生成可用结果'
  }

  return `已完成 ${batch.completed_count} / ${batch.file_count} 个文件`
})

const batchFocusLead = computed(() => {
  const item = batchFocusItem.value
  const batch = currentBatchView.value
  if (!item || !batch) {
    return ''
  }

  const ordinal =
    batchFocusIndex.value >= 0 ? `第 ${batchFocusIndex.value + 1}/${batch.file_count} 个文件` : '当前文件'

  if (batchActiveItem.value?.item_id === item.item_id) {
    return `正在处理 ${ordinal}`
  }
  if (item.status === 'failed') {
    return `最近失败文件 · ${ordinal}`
  }
  if (item.status === 'completed') {
    return `最近完成文件 · ${ordinal}`
  }

  return `等待处理 · ${ordinal}`
})

const batchFocusMeta = computed(() => {
  const item = batchFocusItem.value
  if (!item) {
    return []
  }

  const meta = [`阶段：${getBatchStageLabel(item.status)}`]
  if (item.output_filename) {
    meta.push(`输出：${item.output_filename}`)
  } else if (item.error_message) {
    meta.push(item.error_message)
  } else if (item.message) {
    meta.push(item.message)
  }

  return meta
})

const batchFocusSteps = computed(() => {
  const item = batchFocusItem.value
  const currentStepIndex = resolveBatchStepIndex(item)
  const isFailed = item?.status === 'failed'
  const isCompleted = item?.status === 'completed'

  return BATCH_STEP_SEQUENCE.map((step, index) => ({
    key: step.key,
    label: isFailed && step.key === 'completed' ? '终止' : step.label,
    done: isCompleted ? true : index < currentStepIndex,
    current: !!item && !isCompleted && index === currentStepIndex,
    error: isFailed && index === currentStepIndex
  }))
})

const batchExecutionSummary = computed(() => {
  const batch = currentBatchView.value
  if (!batch) {
    return ''
  }
  const outputFolderText = batch.output_folder_name
    ? `输出镜像文件夹为 ${batch.output_folder_name}。`
    : ''
  return `文件夹 ${batch.folder_name}，共 ${batch.file_count} 个文件，已完成 ${batch.completed_count} 个，成功 ${batch.succeeded_count} 个，失败 ${batch.failed_count} 个。${outputFolderText}`
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

const analysisReviewSummary = computed(() => {
  const metadata = analysisResult.value?.metadata || {}
  if (metadata.recognition_profile !== 'high_quality_lowmem') {
    return ''
  }
  return buildReviewSummary(metadata)
})

const analysisReviewAlertType = computed(() => {
  const metadata = analysisResult.value?.metadata || {}
  if (metadata.requires_manual_review || metadata.review_error || metadata.review_model_installed === false) {
    return 'warning'
  }
  if (metadata.review_model_used) {
    return 'success'
  }
  return 'info'
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

  if (metadata.recognition_profile === 'high_quality_lowmem') {
    return ''
  } else if (metadata.engine_strategy === 'precision_4b') {
    parts.push('当前使用传统 4B 策略')
  } else if (metadata.engine_strategy === 'review_27b') {
    parts.push('当前使用 27B 精审策略')
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

const analysisOcrSummary = computed(() => {
  const metadata = analysisResult.value?.metadata || {}
  const warnings = Array.isArray(metadata.warnings) ? metadata.warnings : []
  const ocrPages = Number(metadata.ocr_pages || 0)
  const totalPages = Number(metadata.pages || 0)
  const effectiveOcrModel = String(
    metadata.ocr_upgrade_to_model || metadata.effective_ocr_model || metadata.ocr_model || ''
  ).trim()
  const mainModel = String(
    analysisResult.value?.llm_model || metadata.requested_llm_model || ''
  ).trim()

  if (warnings.includes('ocr_model_missing')) {
    return 'PDF OCR 模型缺失，请先下载 RapidOCR PP-OCRv5 mobile det/rec 模型后重试。'
  }

  if (warnings.includes('ocr_quality_gate_failed')) {
    const lowPages = Array.isArray(metadata.ocr_quality_gate_failed_pages)
      ? metadata.ocr_quality_gate_failed_pages.join('、')
      : ''
    return lowPages
      ? `PDF OCR 质量门禁未通过，第 ${lowPages} 页需要对照原 PDF 复核。`
      : 'PDF OCR 质量门禁未通过，建议对照原 PDF 复核低质量页。'
  }

  if (warnings.includes('ocr_low_quality_pages')) {
    const lowPages = Array.isArray(metadata.ocr_low_quality_pages)
      ? metadata.ocr_low_quality_pages.join('、')
      : ''
    return lowPages
      ? `PDF OCR 检测到低质量页：第 ${lowPages} 页，建议人工复核。`
      : 'PDF OCR 检测到低质量页，建议人工复核。'
  }

  if (metadata.ocr_quality_gate === 'review_required') {
    return 'PDF OCR 已完成，但部分页面达到复核阈值，建议重点查看中等质量页和低置信行。'
  }

  if (ocrPages > 0 && metadata.ocr_upgrade_applied && effectiveOcrModel) {
    const pageText = totalPages > 0 ? `扫描页 ${ocrPages}/${totalPages} 页` : `扫描页 ${ocrPages} 页`
    const mainText = mainModel ? `；主识别模型仍为 ${mainModel}` : ''
    return `检测到${pageText}，OCR 已自动升级到 ${effectiveOcrModel}${mainText}。`
  }

  if (ocrPages > 0 && effectiveOcrModel) {
    const pageText = totalPages > 0 ? `${ocrPages}/${totalPages} 页扫描内容` : `${ocrPages} 页扫描内容`
    return `文档包含${pageText}，OCR 使用模型 ${effectiveOcrModel}。`
  }

  if (warnings.includes('ocr_timeout_for_scan_pages')) {
    return '文档包含扫描页，但 OCR 在时限内未完成，系统已跳过超时页并继续处理，建议人工复核。'
  }

  if (warnings.includes('ocr_unavailable_for_scan_pages')) {
    if (metadata.recognition_profile === 'high_quality_lowmem') {
      return '文档包含扫描页，但当前 OCR 服务不可用，建议检查本机 OCR 能力和原始 PDF 可读性。'
    }
    return '文档包含扫描页，但当前 OCR 服务不可用，建议检查 Ollama 与可用的 27B 模型。'
  }

  return ''
})

const analysisOcrAlertType = computed(() => {
  const metadata = analysisResult.value?.metadata || {}
  const warnings = Array.isArray(metadata.warnings) ? metadata.warnings : []
  if (warnings.includes('ocr_timeout_for_scan_pages')) {
    return 'warning'
  }
  if (warnings.includes('ocr_unavailable_for_scan_pages')) {
    return 'warning'
  }
  if (
    warnings.includes('ocr_model_missing') ||
    warnings.includes('ocr_quality_gate_failed') ||
    warnings.includes('ocr_low_quality_pages') ||
    metadata.ocr_quality_gate === 'review_required'
  ) {
    return 'warning'
  }
  return metadata.ocr_upgrade_applied ? 'success' : 'info'
})

const analysisWorkflowRecommendation = computed(() => {
  const metadata = analysisResult.value?.metadata || {}
  const recommendedModel = String(metadata.recommended_llm_model || '').trim()
  const mainModel = String(analysisResult.value?.llm_model || '').trim()
  if (!recommendedModel || recommendedModel === mainModel) {
    return ''
  }

  const reason = String(metadata.recommended_llm_reason || '').trim()
  return reason || `当前文档存在扫描内容，若需要进一步精查，建议后续整篇切换到 ${recommendedModel}。`
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

  if (metadata.recognition_profile === 'high_quality_lowmem') {
    qualityParts.push('已启用高质量低内存闭环')
  } else if (metadata.engine_strategy === 'precision_4b') {
    qualityParts.push('已启用 4B 传统闭环')
  } else if (metadata.engine_strategy === 'review_27b') {
    qualityParts.push('已启用 27B 精审闭环')
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
      qualityGateReason ? `质量检查提示：${qualityGateReason}` : '质量检查提示'
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

    if (metadata.recognition_profile === 'high_quality_lowmem') {
      return buildReviewSummary(metadata) || '小 Qwen：未参与。'
    }

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

const supportsSystemNotification = () => typeof window !== 'undefined' && 'Notification' in window

const ensureNotificationPermission = async () => {
  if (!supportsSystemNotification()) {
    return 'denied'
  }
  if (Notification.permission !== 'default') {
    return Notification.permission
  }
  try {
    return await Notification.requestPermission()
  } catch {
    return Notification.permission
  }
}

const notifySystem = async (title: string, body: string) => {
  const permission = await ensureNotificationPermission()
  if (permission !== 'granted') {
    return
  }

  try {
    new Notification(title, {
      body,
      tag: 'contract-desensitize-status'
    })
  } catch {
    // Ignore notification failures in unsupported browser contexts.
  }
}

const resetSingleState = () => {
  analysisPoller.stop()
  processPoller.stop()
  currentFile.value = null
  analysisPending.value = false
  analysisTaskStatus.value = null
  analysisResult.value = null
  desensitizeResult.value = null
  resultDialogVisible.value = false
  activeTab.value = 'entities'
}

const resetBatchState = () => {
  batchPoller.stop()
  currentFolderName.value = ''
  currentFolderFiles.value = []
  batchPending.value = false
  batchTaskStatus.value = null
  batchResult.value = null
  if (folderInputRef.value) {
    folderInputRef.value.value = ''
  }
}

const batchPoller = useTaskPolling<BatchTaskStatus, BatchResult>({
  fetchStatus: getBatchTaskStatus,
  fetchResult: getBatchResult,
  getStatus: (status) => status.status,
  isReady: (status) => status.status === 'completed',
  onStatus: (status) => {
    batchTaskStatus.value = status
  },
  onResult: (result, status) => {
    batchResult.value = result
    batchTaskStatus.value = status
    batchPending.value = false
    ElMessage.success('批量脱敏完成')
    void notifySystem(
      '合同脱敏系统',
      `${result.folder_name} 处理完成，成功 ${result.succeeded_count} 个文件。`
    )
  },
  onFailed: (status) => {
    batchPending.value = false
    ElMessage.error(status.error_message || status.message || '批量处理失败')
  },
  onError: () => {
    batchPending.value = false
  }
})

const analysisPoller = useTaskPolling<TaskStatus, AnalyzeResponse>({
  fetchStatus: getTaskStatus,
  fetchResult: getAnalyzeResult,
  getStatus: (status) => status.status,
  isReady: (status) => ['ready', 'processing', 'anonymizing', 'completed'].includes(status.status),
  onStatus: (status) => {
    analysisTaskStatus.value = status
  },
  onResult: (result, status) => {
    applyAnalysisResult(result)
    analysisPending.value = false
    ElMessage.success('识别完成')
    void notifySystem(
      '合同脱敏系统',
      `${status.filename || currentFile.value?.name || '当前文档'} 识别完成，可以继续生成脱敏文件。`
    )
  },
  onFailed: (status) => {
    analysisPending.value = false
    ElMessage.error(status.error_message || status.message || '识别失败')
  },
  onError: () => {
    analysisPending.value = false
  }
})

const processPoller = useTaskPolling<TaskStatus, DesensitizeResponse>({
  fetchStatus: getTaskStatus,
  fetchResult: getProcessedResult,
  getStatus: (status) => status.status,
  isReady: (status) => status.status === 'completed',
  onStatus: (status) => {
    analysisTaskStatus.value = status
  },
  onResult: (result, status) => {
    desensitizeResult.value = result
    if (result.entities?.length && analysisResult.value) {
      analysisResult.value.entities = result.entities
    }
    if (result.llm_model) {
      selectedLlmModel.value = result.llm_model
    }
    if (result.anonymization_strategy) {
      selectedAnonymizationStrategy.value = result.anonymization_strategy
    }
    processing.value = false
    resultDialogVisible.value = true
    ElMessage.success('脱敏完成')
    void notifySystem(
      '合同脱敏系统',
      `${result.output_filename || status.filename || currentFile.value?.name || '当前文档'} 已生成，可直接下载。`
    )
  },
  onFailed: (status) => {
    processing.value = false
    ElMessage.error(status.error_message || status.message || '脱敏失败')
  },
  onError: () => {
    processing.value = false
  }
})

const applyAnalysisResult = (result: AnalyzeResponse) => {
  analysisResult.value = result
  if (result.llm_model) {
    selectedLlmModel.value = result.llm_model
  }
  if (result.anonymization_strategy) {
    selectedAnonymizationStrategy.value = result.anonymization_strategy
  }
  activeTab.value = 'entities'
}

const formatTaskStatus = (status: string) => {
  const labels: Record<string, string> = {
    queued: '已排队',
    parsing: '解析中',
    analyzing: '识别中',
    ready: '识别完成',
    processing: '处理中',
    anonymizing: '导出中',
    completed: '已完成',
    failed: '失败'
  }

  return labels[status] || status
}

const getBatchStatusTagType = (status: string) => {
  const typeMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    queued: 'info',
    processing: 'warning',
    parsing: 'info',
    analyzing: 'warning',
    anonymizing: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return typeMap[status] || 'info'
}

const getBatchTableRowClass = ({ row }: { row: BatchFileItem }) => {
  if (batchActiveItem.value?.item_id === row.item_id) {
    return 'batch-processing-row'
  }
  if (row.status === 'failed') {
    return 'batch-failed-row'
  }
  return ''
}

const handleFileChange = (file: UploadFile) => {
  resetBatchState()
  resetSingleState()
  currentFile.value = file.raw instanceof File ? file.raw : null
}

const triggerFolderPicker = () => {
  folderInputRef.value?.click()
}

const handleFolderSelection = (event: Event) => {
  const input = event.target as HTMLInputElement
  const files = Array.from(input.files || [])
  resetSingleState()
  batchPoller.stop()
  batchPending.value = false
  batchTaskStatus.value = null
  batchResult.value = null

  if (!files.length) {
    currentFolderName.value = ''
    currentFolderFiles.value = []
    return
  }

  const supportedExtensions = new Set(['pdf', 'docx', 'txt'])
  const nextFiles = files
    .map((file) => ({
      file,
      relativePath: file.webkitRelativePath || file.name
    }))
    .filter((entry) => {
      const segments = entry.relativePath.split(/[\\/]/).filter(Boolean)
      const filename = segments[segments.length - 1] || entry.file.name
      const extension = filename.includes('.') ? filename.split('.').pop()?.toLowerCase() || '' : ''
      return supportedExtensions.has(extension)
    })

  const skippedCount = files.length - nextFiles.length
  if (skippedCount > 0) {
    ElMessage.info(`已自动忽略 ${skippedCount} 个不支持文件，仅保留 TXT、DOCX、PDF。`)
  }

  if (!nextFiles.length) {
    currentFolderName.value = ''
    currentFolderFiles.value = []
    ElMessage.warning('所选文件夹中没有可处理的 TXT、DOCX、PDF 文件。')
    input.value = ''
    return
  }

  const firstPath = nextFiles[0].relativePath.replace(/\\/g, '/')
  currentFolderName.value = firstPath.includes('/') ? firstPath.split('/')[0] : 'selected-folder'
  currentFolderFiles.value = nextFiles
  input.value = ''
}

const loadDefaultSettings = () => {
  const settings = loadAppSettings()
  selectedDesensitizeMode.value = DEFAULT_DESENSITIZE_MODE
  options.value.use_llm = settings.use_llm_default
  options.value.use_custom = settings.use_custom_default
  selectedLlmModel.value = settings.llm_model_default
  selectedAnonymizationStrategy.value = settings.anonymization_strategy_default
  operatorConfig.value = settings.operator_config || {}
  templateName.value = settings.template_name
}

const syncSelectedModel = (preferredModel?: string | null) => {
  selectedLlmModel.value =
    pickModelOption(preferredModel, { requireInstalled: true }) || selectedLlmModel.value
}

const loadModelCatalog = async () => {
  await loadModelCatalogState(selectedDesensitizeMode.value)
  syncSelectedModel(selectedLlmModel.value)
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
    ElMessage.warning('请先选择按需精审模型')
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
    void ensureNotificationPermission()
    analysisPoller.stop()
    analysisPending.value = false
    analysisTaskStatus.value = null
    analysisResult.value = null
    desensitizeResult.value = null
    resultDialogVisible.value = false

    const task = await uploadAndAnalyze(
      currentFile.value,
      options.value.use_llm,
      options.value.use_custom,
      options.value.use_llm ? selectedLlmModel.value : undefined,
      selectedAnonymizationStrategy.value,
      selectedDesensitizeMode.value,
      pageSessionId
    )
    analysisTaskStatus.value = task
    analysisPending.value = true
    activeTab.value = 'entities'
    ElMessage.info(task.message || '文件已上传，正在后台识别。')
    void analysisPoller.poll(task.task_id)
  } finally {
    analyzing.value = false
  }
}

const startBatchProcessing = async () => {
  if (!currentFolderFiles.value.length) {
    ElMessage.warning('请先选择文件夹')
    return
  }

  if (!runtimeReady.value) {
    ElMessage.warning('当前运行环境尚未就绪，请先完成启动检查。')
    return
  }

  if (options.value.use_llm && !selectedLlmModel.value) {
    ElMessage.warning('请先选择按需精审模型')
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

  batchSubmitting.value = true
  try {
    void ensureNotificationPermission()
    resetSingleState()
    batchPoller.stop()
    batchPending.value = false
    batchTaskStatus.value = null
    batchResult.value = null

    const task = await uploadAndProcessFolder(
      currentFolderFiles.value.map((entry) => entry.file),
      currentFolderFiles.value.map((entry) => entry.relativePath),
      currentFolderName.value || 'selected-folder',
      options.value.use_llm,
      options.value.use_custom,
      options.value.use_llm ? selectedLlmModel.value : undefined,
      selectedAnonymizationStrategy.value,
      hasOperatorConfig.value ? operatorConfig.value : undefined,
      selectedDesensitizeMode.value,
      pageSessionId
    )
    batchTaskStatus.value = task
    batchPending.value = true
    ElMessage.info(task.message || '文件夹已上传，正在后台批量处理。')
    void batchPoller.poll(task.batch_id)
  } finally {
    batchSubmitting.value = false
  }
}

const processCurrentTask = async () => {
  if (!analysisResult.value) {
    return
  }

  processing.value = true
  try {
    void ensureNotificationPermission()
    const freshAnalysis = await getAnalyzeResult(analysisResult.value.task_id)
    applyAnalysisResult(freshAnalysis)
    const task = await processDesensitizeApi({
      task_id: freshAnalysis.task_id,
      entities: freshAnalysis.entities,
      config: hasOperatorConfig.value ? operatorConfig.value : undefined,
      llm_model: freshAnalysis.llm_model || selectedLlmModel.value || undefined,
      anonymization_strategy: selectedAnonymizationStrategy.value,
      desensitize_mode: selectedDesensitizeMode.value,
      page_session_id: pageSessionId
    })
    analysisTaskStatus.value = task
    desensitizeResult.value = null
    ElMessage.info(task.message || '已开始后台脱敏与导出。')
    void processPoller.poll(task.task_id)
  } catch (error) {
    processing.value = false
    throw error
  }
}

const openDownload = () => {
  if (!analysisResult.value) {
    return
  }

  const url = desensitizeResult.value?.download_url || buildDownloadUrl(analysisResult.value.task_id)
  window.open(url, '_blank')
}

const openMappingDownload = () => {
  if (!analysisResult.value) {
    return
  }

  const url =
    desensitizeResult.value?.mapping_download_url ||
    buildMappingDownloadUrl(analysisResult.value.task_id)
  window.open(url, '_blank')
}

const openBatchArchive = () => {
  const batchId = batchResult.value?.batch_id || batchTaskStatus.value?.batch_id
  if (!batchId) {
    return
  }
  const url = batchResult.value?.archive_download_url || buildBatchArchiveUrl(batchId)
  window.open(url, '_blank')
}

const openBatchItem = (item: BatchFileItem) => {
  const batchId = batchResult.value?.batch_id || batchTaskStatus.value?.batch_id
  if (!batchId || !item.item_id) {
    return
  }
  const url = item.download_url || buildBatchItemUrl(batchId, item.item_id)
  window.open(url, '_blank')
}

const openBatchItemMapping = (item: BatchFileItem) => {
  const batchId = batchResult.value?.batch_id || batchTaskStatus.value?.batch_id
  if (!batchId || !item.item_id) {
    return
  }
  const url = item.mapping_download_url || buildBatchItemMappingUrl(batchId, item.item_id)
  window.open(url, '_blank')
}

const getTypeName = (type: string) => {
  const typeMap: Record<string, string> = {
    PERSON: '人名',
    ORGANIZATION: '组织机构',
    LOCATION: '地址',
    GOVERNMENT: '政府机构'
  }

  return typeMap[type] || type
}

const getTagType = (type: string) => {
  const tagMap: Record<string, '' | 'success' | 'warning' | 'danger' | 'info' | 'primary'> = {
    PERSON: 'success',
    ORGANIZATION: 'warning',
    LOCATION: 'info',
    GOVERNMENT: 'primary'
  }

  return tagMap[type] || ''
}

const formatReviewSkipReason = (reason: unknown) => {
  const reasonKey = String(reason || '').trim()
  const reasonMap: Record<string, string> = {
    primary_pipeline_sufficient: '前置识别已经覆盖，不需要补漏',
    no_risk_snippets: '没有发现需要精审的高风险片段',
    review_model_missing: '精审模型未安装',
    review_disabled: '精审功能已关闭'
  }
  return reasonMap[reasonKey] || reasonKey
}

const buildReviewSummary = (metadata: Record<string, any>) => {
  const selectedCount = Number(metadata.review_snippet_count || 0)
  if (metadata.review_model_used) {
    const qwenNew = Number(metadata.qwen_new_entities_after_merge || 0)
    const qwenRejected = Number(metadata.qwen_rejected_entities || 0)
    const qwenConfirmed = Number(metadata.qwen_confirmed_overlaps || 0)
    const rawCount = Number(metadata.qwen_raw_candidates || metadata.qwen_materialized_entities || 0)
    if (rawCount > 0) {
      return `小 Qwen：已参与，精审 ${selectedCount} 段，返回 ${rawCount} 个候选，确认 ${qwenConfirmed} 个，新增 ${qwenNew} 个，否决 ${qwenRejected} 个错误候选。`
    }
    return `小 Qwen：已参与，精审 ${selectedCount} 段，未返回可用候选。`
  }
  if (metadata.review_quality_mode === 'rule_first') {
    return '规则层质量检查已启用，小模型仅在高风险片段中兜底。'
  }
  if (metadata.review_model_installed === false) {
    return '精审模型未参与，模型未安装。'
  }
  const reason = formatReviewSkipReason(metadata.review_skipped_reason)
  if (reason) {
    return `精审模型未参与，${reason}。`
  }
  return ''
}

const getSourceName = (source: string) => {
  const sourceMap: Record<string, string> = {
    regex: '规则',
    custom: '自定义',
    contract: '合同字段',
    contract_structure_backfill: '合同结构回填',
    uie: '中文实体模型',
    ner: '中文 NER',
    secondary_ner: '第二路 NER',
    qwen_fragment_review: 'Qwen 片段精审',
    alias_propagation: '简称传播',
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
    contract_structure_backfill: 'info',
    uie: 'primary',
    ner: 'primary',
    secondary_ner: 'primary',
    qwen_fragment_review: 'warning',
    alias_propagation: 'warning',
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

const loadExistingAnalysis = async (taskId: string) => {
  try {
    const result = await getAnalyzeResult(taskId)
    applyAnalysisResult(result)
    analysisTaskStatus.value = {
      task_id: taskId,
      filename: result.filename,
      status: 'ready',
      progress: 100,
      message: '已载入现有识别结果',
      created_at: new Date().toISOString()
    }
  } catch {
    // Ignore invalid deep link task IDs.
  }
}

onMounted(async () => {
  loadDefaultSettings()
  await loadRuntimeStatusState(selectedDesensitizeMode.value)
  await loadModelCatalog()
  syncSelectedModel(selectedLlmModel.value)
  const taskId = typeof route.query.task_id === 'string' ? route.query.task_id : ''
  if (taskId) {
    await loadExistingAnalysis(taskId)
  }
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

.hidden-folder-input {
  display: none;
}

.folder-picker-card {
  margin-bottom: 16px;
  padding: 14px;
  border-radius: 12px;
  background: #f8fafc;
  border: 1px dashed #cbd5e1;
}

.folder-picker-title {
  font-size: 14px;
  font-weight: 600;
  color: #1f2937;
}

.folder-picker-subtitle {
  margin: 6px 0 12px;
  font-size: 12px;
  line-height: 1.7;
  color: #6b7280;
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

.batch-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.batch-progress-shell {
  padding: 18px;
  border-radius: 16px;
  border: 1px solid #dbe7f5;
  background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
}

.batch-progress-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 16px;
}

.batch-progress-kicker {
  font-size: 12px;
  font-weight: 600;
  color: #64748b;
  letter-spacing: 0.04em;
}

.batch-progress-title {
  margin-top: 6px;
  font-size: 20px;
  font-weight: 700;
  color: #0f172a;
}

.batch-progress-caption {
  margin-top: 8px;
  font-size: 13px;
  line-height: 1.7;
  color: #64748b;
}

.batch-progress-ratio {
  min-width: 104px;
  padding: 14px 16px;
  border-radius: 16px;
  background: #0f172a;
  color: #f8fafc;
  text-align: right;
}

.batch-progress-ratio span {
  display: block;
  font-size: 30px;
  font-weight: 700;
  line-height: 1;
}

.batch-progress-ratio small {
  display: block;
  margin-top: 4px;
  font-size: 12px;
  color: rgba(248, 250, 252, 0.72);
}

.batch-progress-bar {
  margin-bottom: 18px;
}

.batch-stats-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.batch-stat {
  padding: 14px;
  border-radius: 14px;
  border: 1px solid #e2e8f0;
  background: rgba(255, 255, 255, 0.88);
}

.batch-stat.is-accent {
  background: #eff6ff;
  border-color: #bfdbfe;
}

.batch-stat.is-success {
  background: #ecfdf3;
  border-color: #bbf7d0;
}

.batch-stat.is-danger {
  background: #fff1f2;
  border-color: #fecdd3;
}

.batch-stat-label {
  font-size: 12px;
  color: #64748b;
}

.batch-stat-value {
  margin-top: 8px;
  font-size: 24px;
  font-weight: 700;
  color: #0f172a;
}

.batch-focus-panel {
  margin-top: 16px;
  padding: 16px;
  border-radius: 14px;
  border: 1px solid #e2e8f0;
  background: rgba(255, 255, 255, 0.95);
}

.batch-focus-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
}

.batch-focus-kicker {
  font-size: 12px;
  font-weight: 600;
  color: #64748b;
}

.batch-focus-path {
  margin-top: 6px;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.6;
  color: #111827;
  word-break: break-word;
}

.batch-focus-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 12px;
  margin: 12px 0;
  font-size: 12px;
  color: #64748b;
}

.batch-step-track {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.batch-step-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  border: 1px solid #e2e8f0;
  background: #f8fafc;
  font-size: 12px;
  color: #64748b;
}

.batch-step-chip.is-done {
  background: #f0fdf4;
  border-color: #bbf7d0;
  color: #166534;
}

.batch-step-chip.is-current {
  background: #eff6ff;
  border-color: #93c5fd;
  color: #1d4ed8;
}

.batch-step-chip.is-error {
  background: #fff1f2;
  border-color: #fda4af;
  color: #be123c;
}

.batch-step-dot {
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: currentColor;
}

.batch-file-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.batch-file-name {
  font-weight: 600;
  line-height: 1.6;
  color: #111827;
  word-break: break-word;
}

.batch-mini-progress {
  min-width: 140px;
}

.batch-mini-progress .secondary-text {
  margin-top: 6px;
}

:deep(.batch-processing-row) {
  --el-table-tr-bg-color: #eff6ff;
}

:deep(.batch-failed-row) {
  --el-table-tr-bg-color: #fff7ed;
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

.statistics-section {
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

.mt-12 {
  margin-top: 12px;
}

@media (max-width: 992px) {
  .panel-card {
    min-height: auto;
  }

  .panel-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .batch-toolbar {
    flex-direction: column;
    align-items: stretch;
  }

  .batch-progress-header {
    flex-direction: column;
  }

  .batch-progress-ratio {
    width: 100%;
    text-align: left;
  }

  .batch-stats-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .batch-focus-header {
    flex-direction: column;
  }
}
</style>
