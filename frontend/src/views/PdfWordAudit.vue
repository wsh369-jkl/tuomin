<template>
  <div class="audit-page">
    <el-row :gutter="20">
      <el-col :xs="24" :lg="8">
        <el-card class="panel-card">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">PDF 转 Word 核查</div>
                <div class="panel-subtitle">
                  上传原始 PDF 和 WPS 转换 DOCX，系统会重新 OCR 并在 WPS 文档上标注差异、证据与人工复核项。
                </div>
              </div>
            </div>
          </template>

          <div class="upload-label">原始 PDF</div>
          <el-upload
            class="upload-area"
            drag
            :auto-upload="false"
            :limit="1"
            accept=".pdf"
            :on-change="handlePdfChange"
            :on-remove="handlePdfRemove"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">选择原始 PDF</div>
          </el-upload>

          <div class="upload-label">WPS 转换 DOCX</div>
          <el-upload
            class="upload-area"
            drag
            :auto-upload="false"
            :limit="1"
            accept=".docx"
            :on-change="handleDocxChange"
            :on-remove="handleDocxRemove"
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">选择 WPS 转换后的 Word 文档</div>
          </el-upload>

          <el-descriptions :column="1" border class="mb-16">
            <el-descriptions-item label="PDF">
              {{ pdfFile?.name || '未选择' }}
            </el-descriptions-item>
            <el-descriptions-item label="WPS DOCX">
              {{ wpsDocxFile?.name || '未选择' }}
            </el-descriptions-item>
          </el-descriptions>

          <el-alert
            class="mb-16"
            type="info"
            :closable="false"
            title="输出文件会保留 WPS 的版面结构和正文内容，只写入 OCR 复核批注；完整证据包可用于人工核查和调参。"
          />

          <el-button
            type="primary"
            size="large"
            class="full-width"
            :loading="pending"
            :disabled="!pdfFile || !wpsDocxFile"
            @click="startAudit"
          >
            开始核查
          </el-button>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="16">
        <el-card class="panel-card">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">核查结果</div>
                <div class="panel-subtitle">全量 OCR、版面解析和多模型证据融合；默认不改正文，只输出发现项和批注。</div>
              </div>
              <el-tag v-if="auditResult" type="success" size="large">
                {{ auditResult.metadata.comment_count || 0 }} 条批注
              </el-tag>
            </div>
          </template>

          <div v-if="auditStatus && pending" class="mb-16">
            <el-alert type="info" :closable="false" :title="auditStatus.message || '正在核查...'" />
            <el-progress class="mt-12" :percentage="auditStatus.progress" />
          </div>

          <el-empty v-if="!auditStatus && !auditResult" description="请先上传 PDF 和 WPS DOCX 后开始核查。" />

          <template v-else-if="auditResult">
            <el-alert
              class="mb-16"
              :type="productStatusType"
              :closable="false"
              :title="productStatusText"
            />

            <div class="metrics-grid">
              <div class="metric-card">
                <div class="metric-label">确认错误</div>
                <div class="metric-value">{{ resultMetrics.confirmed_count }}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">疑似错误</div>
                <div class="metric-value">{{ resultMetrics.suspected_count }}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">模型冲突</div>
                <div class="metric-value">{{ resultMetrics.model_conflict_count }}</div>
              </div>
              <div class="metric-card">
                <div class="metric-label">覆盖不足</div>
                <div class="metric-value">{{ resultMetrics.coverage_gap_count }}</div>
              </div>
            </div>

            <div class="coverage-grid">
              <div class="coverage-card">
                <span>页数</span>
                <strong>{{ productSummary.page_count || auditResult.metadata.page_count || 0 }}</strong>
              </div>
              <div class="coverage-card">
                <span>人工复核任务</span>
                <strong>{{ productSummary.human_review_task_count || 0 }}</strong>
              </div>
              <div class="coverage-card">
                <span>表格未闭环单元</span>
                <strong>{{ tableSummary.unresolved_cell_count || 0 }}</strong>
              </div>
              <div class="coverage-card">
                <span>覆盖未闭环</span>
                <strong>{{ coverageSummary.unresolved_count || 0 }}</strong>
              </div>
              <div class="coverage-card">
                <span>模型守门降级</span>
                <strong>{{ modelGuardSummary.guarded_count || 0 }}</strong>
              </div>
              <div class="coverage-card">
                <span>峰值内存</span>
                <strong>{{ formatMemory(auditResult.metadata.peak_memory_mib || auditResult.metadata.audit_pipeline_peak_mib) }}</strong>
              </div>
            </div>

            <div class="result-actions">
              <el-button type="primary" :icon="Download" @click="downloadAuditedDocx">
                下载复核 DOCX
              </el-button>
              <el-button :icon="Download" @click="downloadReport">
                下载 JSON 报告
              </el-button>
              <el-button :icon="Download" @click="downloadEvidence">
                下载证据包
              </el-button>
            </div>

            <el-alert
              v-if="warningsText"
              class="mb-16"
              type="warning"
              :closable="false"
              :title="warningsText"
            />

            <div class="finding-filters">
              <el-select v-model="severityFilter" clearable placeholder="Severity" size="small">
                <el-option v-for="item in severityOptions" :key="item" :label="item" :value="item" />
              </el-select>
              <el-select v-model="categoryFilter" clearable placeholder="Category" size="small">
                <el-option v-for="item in categoryOptions" :key="item" :label="item" :value="item" />
              </el-select>
              <el-select v-model="sourceFilter" clearable placeholder="Evidence source" size="small">
                <el-option v-for="item in sourceOptions" :key="item" :label="item" :value="item" />
              </el-select>
            </div>

            <el-tabs v-model="activeTab">
              <el-tab-pane label="全部" name="all">
                <finding-list :items="allFilteredFindings" empty-text="没有符合筛选条件的发现项。" />
              </el-tab-pane>
              <el-tab-pane label="确认错误" name="confirmed">
                <finding-list :items="confirmedFindings" empty-text="没有确认错误。" />
              </el-tab-pane>
              <el-tab-pane label="疑似错误" name="suspected">
                <finding-list :items="suspectedFindings" empty-text="没有疑似错误。" />
              </el-tab-pane>
              <el-tab-pane label="模型冲突" name="conflict">
                <finding-list :items="conflictFindings" empty-text="没有模型冲突。" />
              </el-tab-pane>
              <el-tab-pane label="覆盖不足" name="coverage">
                <finding-list :items="coverageFindings" empty-text="没有覆盖不足项。" />
              </el-tab-pane>
              <el-tab-pane label="页级风险" name="pages">
                <div v-if="pageRiskSummary.length" class="page-risk-grid">
                  <div
                    v-for="page in pageRiskSummary"
                    :key="page.page_no"
                    class="page-risk-card"
                    :class="`risk-${page.risk_level || 'low'}`"
                  >
                    <div class="page-risk-title">
                      <strong>第 {{ page.page_no }} 页</strong>
                      <span>{{ page.risk_level || 'low' }}</span>
                    </div>
                    <div class="page-risk-terminal">
                      终态：{{ page.terminal_label || page.terminal_state || '未判定' }}
                      <span v-if="page.open_issue_count"> / 开放问题 {{ page.open_issue_count }}</span>
                    </div>
                    <div class="page-risk-stats">
                      确认 {{ page.confirmed_count || 0 }} / 疑似 {{ page.suspected_count || 0 }} /
                      冲突 {{ page.model_conflict_count || 0 }} / 任务 {{ page.review_task_count || 0 }}
                    </div>
                    <div class="page-risk-reasons">
                      {{ page.terminal_reason || '暂无终态说明' }}
                    </div>
                    <div class="page-risk-reasons secondary">
                      {{ formatList(page.reasons) || formatList(page.labels) || '暂无风险标签' }}
                    </div>
                  </div>
                </div>
                <div v-else class="empty-inline">没有页级风险摘要。</div>
              </el-tab-pane>
              <el-tab-pane label="人工复核" name="queue">
                <div v-if="humanReviewQueue.length" class="review-queue">
                  <div v-for="task in humanReviewQueue" :key="task.task_id" class="review-task">
                    <div class="review-task-title">
                      <strong>第 {{ task.page_no || '-' }} 页</strong>
                      <span>{{ task.task_type }} / {{ task.status }}</span>
                    </div>
                    <div class="review-task-reason">{{ task.reason || '需人工复核。' }}</div>
                    <div class="review-task-route">后续：{{ task.next_engine || task.next_route || '-' }}</div>
                  </div>
                </div>
                <div v-else class="empty-inline">没有人工复核队列。</div>
              </el-tab-pane>
              <el-tab-pane label="表格/覆盖" name="product">
                <div class="product-summary-grid">
                  <div class="product-summary-card">
                    <span>表格已审单元</span>
                    <strong>{{ tableSummary.reviewed_cell_count || 0 }}</strong>
                  </div>
                  <div class="product-summary-card">
                    <span>表格确认/疑似</span>
                    <strong>{{ tableSummary.confirmed_cell_count || 0 }}/{{ tableSummary.suspected_cell_count || 0 }}</strong>
                  </div>
                  <div class="product-summary-card">
                    <span>覆盖审查项</span>
                    <strong>{{ coverageSummary.coverage_review_count || 0 }}</strong>
                  </div>
                  <div class="product-summary-card">
                    <span>高风险表格页</span>
                    <strong>{{ formatList(tableSummary.high_risk_pages) || '-' }}</strong>
                  </div>
                </div>
              </el-tab-pane>
            </el-tabs>
          </template>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed, defineComponent, h, ref } from 'vue'
import type { UploadFile } from 'element-plus'
import { ElMessage } from 'element-plus'
import { Download, UploadFilled } from '@element-plus/icons-vue'
import {
  downloadPdfWordAuditEvidence,
  downloadPdfWordAuditReport,
  downloadPdfWordAuditResult,
  getPdfWordAuditResult,
  getPdfWordAuditStatus,
  type PdfWordAuditFinding,
  type PdfWordAuditResult,
  type PdfWordAuditStatus,
  uploadForPdfWordAudit
} from '@/api/pdfWordAudit'
import { useTaskPolling } from '@/composables/useTaskPolling'

const pdfFile = ref<File | null>(null)
const wpsDocxFile = ref<File | null>(null)
const pending = ref(false)
const auditStatus = ref<PdfWordAuditStatus | null>(null)
const auditResult = ref<PdfWordAuditResult | null>(null)
const activeTab = ref('all')
const severityFilter = ref('')
const categoryFilter = ref('')
const sourceFilter = ref('')

const productReport = computed<Record<string, any>>(() => auditResult.value?.product_report || {})
const productSummary = computed<Record<string, any>>(() => productReport.value.summary || {})
const tableSummary = computed<Record<string, any>>(() => auditResult.value?.table_summary || productReport.value.table_summary || {})
const coverageSummary = computed<Record<string, any>>(() => auditResult.value?.coverage_summary || productReport.value.coverage_summary || {})
const modelGuardSummary = computed<Record<string, any>>(() => productReport.value.model_guard_summary || {})
const pageRiskSummary = computed<Record<string, any>[]>(() =>
  auditResult.value?.page_risk_summary || productReport.value.page_risk_summary || []
)
const humanReviewQueue = computed<Record<string, any>[]>(() =>
  auditResult.value?.human_review_queue || productReport.value.human_review_queue || []
)
const productStatusText = computed(() => {
  const status = productReport.value.status || 'unknown'
  const map: Record<string, string> = {
    needs_human_review: '当前结果仍有覆盖不足、模型冲突或人工复核任务，不能当作全自动确认结果。',
    confirmed_errors_found: '已发现确认错误；DOCX 只写批注，正文未被自动修改。',
    no_confirmed_error: '未发现确认错误；仍建议检查覆盖不足和人工复核队列。',
    unknown: '尚未生成产品化报告摘要。'
  }
  return map[status] || `产品状态：${status}`
})
const productStatusType = computed(() => {
  const status = productReport.value.status
  if (status === 'confirmed_errors_found') return 'success'
  if (status === 'no_confirmed_error') return 'info'
  return 'warning'
})

const correctionToFinding = (item: PdfWordAuditResult['corrections'][number]): PdfWordAuditFinding => {
  const title = (item.comment_text || '').split(/\r?\n/)[0] || item.reason || 'DOCX 批注'
  const confirmed = title.includes('确认错误') || item.comment_text.includes('必须改为')
  return {
    id: item.id,
    severity: confirmed ? 'high' : 'medium',
    category: title.includes('表格') ? 'table_cell_mismatch' : 'substitution',
    page_no: item.page_no,
    wps_text: item.old_text,
    suggested_text: item.new_text,
    diff_ops: item.old_text === item.new_text ? [] : [{ op: 'replace', old_text: item.old_text, new_text: item.new_text }],
    confidence: item.confidence || 0,
    status: confirmed ? 'confirmed_error' : 'suspected_error',
    evidence_sources: ['docx_comment'],
    bbox_refs: [],
    crop_refs: [],
    wps_anchor: { unit_id: item.wps_unit_id, correction_id: item.id, action: item.action },
    reason: item.reason || title,
    requires_human_review: true
  }
}

const allFindings = computed(() => {
  const findings = auditResult.value?.findings || []
  if (findings.length) return findings
  return (auditResult.value?.corrections || []).map(correctionToFinding)
})
const resultMetrics = computed(() => {
  const counts = {
    confirmed_count: 0,
    suspected_count: 0,
    model_conflict_count: 0,
    coverage_gap_count: 0
  }
  for (const item of allFindings.value) {
    if (item.status === 'confirmed_error') counts.confirmed_count += 1
    else if (item.status === 'suspected_error') counts.suspected_count += 1
    else if (item.status === 'model_conflict') counts.model_conflict_count += 1
    else if (item.status === 'coverage_gap') counts.coverage_gap_count += 1
  }
  const metadata = auditResult.value?.metadata || {}
  return {
    confirmed_count: counts.confirmed_count || Number(metadata.confirmed_count || 0),
    suspected_count: counts.suspected_count || Number(metadata.suspected_count || 0),
    model_conflict_count: counts.model_conflict_count || Number(metadata.model_conflict_count || 0),
    coverage_gap_count: counts.coverage_gap_count || Number(metadata.coverage_gap_count || 0)
  }
})
const severityOptions = computed(() =>
  Array.from(new Set(allFindings.value.map((item) => item.severity).filter(Boolean))).sort()
)
const categoryOptions = computed(() =>
  Array.from(new Set(allFindings.value.map((item) => item.category).filter(Boolean))).sort()
)
const sourceOptions = computed(() =>
  Array.from(new Set(allFindings.value.flatMap((item) => item.evidence_sources || []).filter(Boolean))).sort()
)

const filterFindings = (items: PdfWordAuditFinding[]) =>
  items.filter((item) => {
    if (severityFilter.value && item.severity !== severityFilter.value) return false
    if (categoryFilter.value && item.category !== categoryFilter.value) return false
    if (sourceFilter.value && !(item.evidence_sources || []).includes(sourceFilter.value)) return false
    return true
  })

const allFilteredFindings = computed(() => filterFindings(allFindings.value))
const confirmedFindings = computed(() =>
  filterFindings(allFindings.value.filter((item) => item.status === 'confirmed_error'))
)
const conflictFindings = computed(() =>
  filterFindings(allFindings.value.filter((item) => item.status === 'model_conflict'))
)
const coverageFindings = computed(() =>
  filterFindings(allFindings.value.filter((item) => item.status === 'coverage_gap'))
)
const suspectedFindings = computed(() =>
  filterFindings(allFindings.value.filter((item) => item.status === 'suspected_error'))
)
const warningsText = computed(() => {
  const warnings = auditResult.value?.metadata?.warnings
  return Array.isArray(warnings) && warnings.length ? `核查提示：${warnings.join('、')}` : ''
})

const formatList = (value: unknown) => {
  if (!Array.isArray(value)) return ''
  return value.filter((item) => item !== null && item !== undefined && String(item).trim()).slice(0, 8).join('、')
}

const formatMemory = (value: unknown) => {
  const number = Number(value || 0)
  return number > 0 ? `${number.toFixed(0)} MiB` : '-'
}

const poller = useTaskPolling<PdfWordAuditStatus, PdfWordAuditResult>({
  fetchStatus: getPdfWordAuditStatus,
  fetchResult: getPdfWordAuditResult,
  getStatus: (status) => status.status,
  isReady: (status) => status.status === 'completed',
  onStatus: (status) => {
    auditStatus.value = status
  },
  onResult: (result) => {
    auditResult.value = result
    pending.value = false
    ElMessage.success('PDF 转 Word 核查完成')
  },
  onFailed: (status) => {
    pending.value = false
    ElMessage.error(status.error_message || status.message || 'PDF 转 Word 核查失败')
  },
  onError: () => {
    pending.value = false
  }
})

const handlePdfChange = (file: UploadFile) => {
  poller.stop()
  pdfFile.value = file.raw instanceof File ? file.raw : null
  auditStatus.value = null
  auditResult.value = null
}

const handlePdfRemove = () => {
  poller.stop()
  pdfFile.value = null
  auditStatus.value = null
  auditResult.value = null
}

const handleDocxChange = (file: UploadFile) => {
  poller.stop()
  wpsDocxFile.value = file.raw instanceof File ? file.raw : null
  auditStatus.value = null
  auditResult.value = null
}

const handleDocxRemove = () => {
  poller.stop()
  wpsDocxFile.value = null
  auditStatus.value = null
  auditResult.value = null
}

const startAudit = async () => {
  if (!pdfFile.value || !wpsDocxFile.value) {
    ElMessage.warning('请同时选择原始 PDF 和 WPS 转换 DOCX')
    return
  }
  pending.value = true
  auditResult.value = null
  try {
    const task = await uploadForPdfWordAudit(pdfFile.value, wpsDocxFile.value)
    auditStatus.value = task
    void poller.poll(task.audit_id)
  } catch {
    pending.value = false
  }
}

const downloadAuditedDocx = () => {
  if (!auditResult.value) {
    return
  }
  window.open(downloadPdfWordAuditResult(auditResult.value.audit_id), '_blank')
}

const downloadReport = () => {
  if (!auditResult.value) {
    return
  }
  window.open(downloadPdfWordAuditReport(auditResult.value.audit_id), '_blank')
}

const downloadEvidence = () => {
  if (!auditResult.value) {
    return
  }
  window.open(downloadPdfWordAuditEvidence(auditResult.value.audit_id), '_blank')
}

const FindingList = defineComponent({
  props: {
    items: {
      type: Array as () => PdfWordAuditFinding[],
      required: true
    },
    emptyText: {
      type: String,
      required: true
    }
  },
  setup(props) {
    return () =>
      props.items.length
        ? h(
            'div',
            { class: 'correction-list' },
            props.items.map((item) =>
              h('div', { class: 'correction-item', key: item.id }, [
                h('div', { class: 'correction-title' }, [
                  h('span', `第 ${item.page_no || '-'} 页`),
                  h('span', `${item.severity} / ${item.category} / 置信 ${item.confidence.toFixed(2)}`)
                ]),
                h('div', { class: 'correction-text old' }, `WPS：${item.wps_text}`),
                h('div', { class: 'correction-text new' }, `模型建议：${item.suggested_text || '（无稳定模型文本）'}`),
                item.crop_refs.length
                  ? h('div', { class: 'correction-crops' }, `截图证据：${item.crop_refs.join('、')}`)
                  : null,
                h('div', { class: 'correction-reason' }, `证据：${item.evidence_sources.join('、') || '无'}；${item.reason}`)
              ])
            )
          )
        : h('div', { class: 'empty-inline' }, props.emptyText)
  }
})
</script>

<style scoped>
.audit-page {
  max-width: 1500px;
}

.panel-card {
  min-height: 760px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  gap: 12px;
}

.panel-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
}

.panel-subtitle {
  color: #6b7280;
  font-size: 13px;
  line-height: 1.6;
}

.upload-label {
  margin: 10px 0 8px;
  font-weight: 600;
  color: #1f2937;
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

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.coverage-grid {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 16px;
}

.coverage-card {
  border: 1px solid #dbeafe;
  border-radius: 8px;
  padding: 10px;
  background: #eff6ff;
  color: #1e3a8a;
}

.coverage-card span,
.coverage-card strong {
  display: block;
}

.coverage-card span {
  font-size: 12px;
  color: #475569;
}

.coverage-card strong {
  margin-top: 4px;
  font-size: 16px;
}

.metric-card {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 12px;
  background: #f8fafc;
}

.metric-label {
  color: #64748b;
  font-size: 12px;
}

.metric-value {
  margin-top: 6px;
  font-size: 24px;
  font-weight: 700;
  color: #111827;
}

.result-actions {
  margin-bottom: 16px;
}

.finding-filters {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 14px;
}

.correction-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.correction-item {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 12px;
  background: #fff;
}

.correction-title {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  font-size: 12px;
  color: #64748b;
}

.correction-text {
  margin-top: 8px;
  line-height: 1.7;
  word-break: break-word;
}

.correction-text.old {
  color: #991b1b;
}

.correction-text.new {
  color: #166534;
}

.correction-reason,
.correction-crops,
.empty-inline {
  margin-top: 8px;
  color: #64748b;
  font-size: 13px;
}

.page-risk-grid,
.review-queue,
.product-summary-grid {
  display: grid;
  gap: 10px;
}

.page-risk-grid {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.page-risk-card,
.review-task,
.product-summary-card {
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 12px;
  background: #fff;
}

.page-risk-card.risk-high {
  border-color: #fecaca;
  background: #fef2f2;
}

.page-risk-card.risk-medium {
  border-color: #fed7aa;
  background: #fff7ed;
}

.page-risk-card.risk-low {
  border-color: #bbf7d0;
  background: #f0fdf4;
}

.page-risk-title,
.review-task-title {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  color: #111827;
}

.page-risk-terminal {
  margin-top: 6px;
  color: #1f2937;
  font-size: 13px;
  font-weight: 600;
  line-height: 1.6;
}

.page-risk-stats,
.page-risk-reasons,
.review-task-reason,
.review-task-route,
.product-summary-card span {
  margin-top: 6px;
  color: #64748b;
  font-size: 13px;
  line-height: 1.6;
}

.page-risk-reasons.secondary {
  opacity: 0.86;
}

.product-summary-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.product-summary-card strong {
  display: block;
  margin-top: 6px;
  color: #111827;
  font-size: 18px;
}

@media (max-width: 992px) {
  .panel-card {
    min-height: auto;
  }

  .metrics-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .coverage-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .finding-filters {
    grid-template-columns: 1fr;
  }

  .page-risk-grid,
  .product-summary-grid {
    grid-template-columns: 1fr;
  }
}
</style>
