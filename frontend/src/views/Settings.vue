<template>
  <div class="settings">
    <el-card class="mb-20">
      <template #header>
        <div class="section-header">
          <div>
            <div class="section-title">默认识别与脱敏配置</div>
            <div class="section-subtitle">这些设置会作为新任务的默认值，也可保存为模板复用</div>
          </div>
          <div class="header-actions">
            <el-button @click="resetSettings">恢复默认</el-button>
            <el-button type="primary" @click="saveSettings">保存设置</el-button>
          </div>
        </div>
      </template>

      <el-form label-width="160px">
        <el-form-item label="默认启用大模型识别">
          <el-switch v-model="settings.use_llm_default" />
        </el-form-item>
        <el-form-item label="默认识别模型">
          <el-select
            v-model="settings.llm_model_default"
            class="full-width"
            placeholder="请选择默认模型"
            :disabled="!modelOptions.length"
          >
            <el-option
              v-for="model in modelOptions"
              :key="model.name"
              :label="formatModelLabel(model)"
              :value="model.name"
              :disabled="modelCatalog?.backend === 'ollama' && !model.installed"
            />
          </el-select>
          <div class="form-tip">
            {{ selectedModelDescription || '新任务会默认使用这个模型，识别页里仍然可以在开始前临时切换。' }}
          </div>
        </el-form-item>
        <el-form-item label="默认启用自定义规则">
          <el-switch v-model="settings.use_custom_default" />
        </el-form-item>
        <el-form-item label="默认隐名策略">
          <el-radio-group v-model="settings.anonymization_strategy_default">
            <el-radio-button
              v-for="option in anonymizationStrategyOptions"
              :key="option.value"
              :label="option.value"
            >
              {{ option.label }}
            </el-radio-button>
          </el-radio-group>
          <div class="form-tip">
            {{ selectedAnonymizationDescription || '默认控制人物和主体名称的替换表达方式。' }}
          </div>
        </el-form-item>
        <el-form-item label="当前模板">
          <div class="template-chip">
            <el-tag v-if="settings.template_name" type="success">
              {{ settings.template_name }}
            </el-tag>
            <span v-else class="muted-text">未绑定模板</span>
          </div>
        </el-form-item>
        <el-form-item label="高级脱敏配置 JSON">
          <el-input
            v-model="operatorConfigText"
            type="textarea"
            :rows="10"
            placeholder='例如：{ "CN_PHONE": { "operator": "replace", "params": { "new_value": "[手机号]" } } }'
          />
          <div class="form-tip">
            这里填写传给后端脱敏引擎的 operator_config。留空或写 {} 表示使用系统默认策略。
          </div>
        </el-form-item>
      </el-form>
    </el-card>

    <el-card class="mb-20">
      <template #header>
        <div class="section-header">
          <div>
            <div class="section-title">配置模板</div>
            <div class="section-subtitle">把当前设置保存为模板，并在不同项目间快速切换</div>
          </div>
          <el-button type="primary" @click="openTemplateDialog">
            <el-icon><Plus /></el-icon>
            保存为模板
          </el-button>
        </div>
      </template>

      <el-table :data="templates" empty-text="暂无配置模板">
        <el-table-column prop="name" label="模板名称" min-width="180">
          <template #default="{ row }">
            <div class="template-name">
              <span>{{ row.name }}</span>
              <el-tag v-if="row.is_default" type="warning" size="small">默认</el-tag>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="说明" min-width="220" />
        <el-table-column label="更新时间" width="180">
          <template #default="{ row }">
            {{ formatDate(row.updated_at) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="220">
          <template #default="{ row }">
            <el-button type="primary" link @click="applyTemplate(row)">
              应用
            </el-button>
            <el-button type="info" link @click="showTemplatePreview(row)">
              预览
            </el-button>
            <el-button type="danger" link @click="removeTemplate(row)">
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-card>
      <template #header>
        <div class="section-header">
          <div>
            <div class="section-title">运行时信息</div>
            <div class="section-subtitle">用于确认当前引擎、模型和支持的实体类型是否正常加载</div>
          </div>
          <el-button @click="loadRuntimeInfo">
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </template>

      <el-descriptions :column="2" border class="mb-20">
        <el-descriptions-item label="系统版本">
          {{ engineInfo?.version ? `v${engineInfo.version}` : '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="架构">
          {{ engineInfo?.architecture || '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="LLM 后端">
          {{ engineInfo?.llm_backend || '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="模型">
          {{ engineInfo?.llm_model || '-' }}
        </el-descriptions-item>
        <el-descriptions-item label="API 地址">
          {{ apiBaseUrl }}
        </el-descriptions-item>
        <el-descriptions-item label="前端地址">
          {{ clientBaseUrl }}
        </el-descriptions-item>
      </el-descriptions>

      <el-divider content-position="left">支持的实体类型</el-divider>
      <div class="tag-group mb-20">
        <el-tag
          v-for="entity in engineInfo?.supported_entities || []"
          :key="entity"
          class="mr-8 mb-8"
        >
          {{ entity }}
        </el-tag>
        <span v-if="!(engineInfo?.supported_entities || []).length" class="muted-text">暂无数据</span>
      </div>

      <el-divider content-position="left">支持的脱敏操作</el-divider>
      <div class="tag-group">
        <el-tag
          v-for="operator in engineInfo?.supported_operators || []"
          :key="operator"
          type="success"
          class="mr-8 mb-8"
        >
          {{ operator }}
        </el-tag>
        <span v-if="!(engineInfo?.supported_operators || []).length" class="muted-text">暂无数据</span>
      </div>
    </el-card>

    <el-dialog v-model="templateDialogVisible" title="保存为模板" width="560px">
      <el-form :model="templateForm" label-width="90px">
        <el-form-item label="模板名称">
          <el-input v-model="templateForm.name" placeholder="例如：合同基础方案" />
        </el-form-item>
        <el-form-item label="模板说明">
          <el-input
            v-model="templateForm.description"
            type="textarea"
            :rows="3"
            placeholder="简要说明这个模板适合什么场景"
          />
        </el-form-item>
        <el-form-item label="设为默认">
          <el-switch v-model="templateForm.is_default" />
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="templateDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="createTemplateFromCurrent">
          保存模板
        </el-button>
      </template>
    </el-dialog>

    <el-drawer
      v-model="templatePreviewVisible"
      title="模板预览"
      size="520px"
    >
      <template v-if="previewTemplate">
        <el-descriptions :column="1" border class="mb-20">
          <el-descriptions-item label="模板名称">
            {{ previewTemplate.name }}
          </el-descriptions-item>
          <el-descriptions-item label="说明">
            {{ previewTemplate.description || '无' }}
          </el-descriptions-item>
        </el-descriptions>
        <pre class="json-preview">{{ previewTemplateJson }}</pre>
      </template>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Refresh } from '@element-plus/icons-vue'
import { getAvailableModels, getEngineInfo } from '@/api/desensitize'
import { createTemplate, deleteTemplate, getTemplates } from '@/api/history'
import type { EngineInfo, LLMModelListResponse, LLMModelOption } from '@/api/desensitize'
import type { ConfigTemplate } from '@/api/history'
import {
  defaultAppSettings,
  loadAppSettings,
  normalizeAppSettings,
  saveAppSettings
} from '@/utils/settings'

const settings = ref(loadAppSettings())
const operatorConfigText = ref(JSON.stringify(settings.value.operator_config, null, 2))
const engineInfo = ref<EngineInfo | null>(null)
const templates = ref<ConfigTemplate[]>([])
const modelCatalog = ref<LLMModelListResponse | null>(null)
const fallbackOrigin = 'http://localhost'
const clientBaseUrl = computed(() =>
  typeof window === 'undefined' ? fallbackOrigin : window.location.origin
)
const apiBaseUrl = computed(() => new URL('/api/v1', clientBaseUrl.value).toString())

const templateDialogVisible = ref(false)
const templatePreviewVisible = ref(false)
const previewTemplate = ref<ConfigTemplate | null>(null)

const templateForm = ref({
  name: '',
  description: '',
  is_default: false
})

const previewTemplateJson = computed(() =>
  JSON.stringify(previewTemplate.value?.config_data ?? {}, null, 2)
)
const anonymizationStrategyOptions = [
  {
    value: 'official',
    label: '官方局部某化',
    description: '保留主体原有阅读感，只对关键名称局部做“某化”，适合正式文书。'
  },
  {
    value: 'serial_roles',
    label: '甲乙丙主体策略',
    description: '人物和主体优先改成甲乙丙类称谓，区分更直观，适合快速阅读。'
  }
]
const modelOptions = computed(() => modelCatalog.value?.models || [])
const selectedModelOption = computed(() =>
  modelOptions.value.find((item) => item.name === settings.value.llm_model_default) || null
)
const selectedModelDescription = computed(() => {
  if (!selectedModelOption.value) {
    return ''
  }
  return `${selectedModelOption.value.strategy_label}：${selectedModelOption.value.strategy_description}`
})
const selectedAnonymizationDescription = computed(() => {
  return (
    anonymizationStrategyOptions.find(
      (item) => item.value === settings.value.anonymization_strategy_default
    )?.description || ''
  )
})

const parseOperatorConfig = () => {
  const text = operatorConfigText.value.trim()
  if (!text) {
    return {}
  }

  const parsed = JSON.parse(text)
  if (!parsed || Array.isArray(parsed) || typeof parsed !== 'object') {
    throw new Error('高级脱敏配置必须是 JSON 对象')
  }
  return parsed as Record<string, any>
}

const buildSettingsPayload = () => {
  const operatorConfig = parseOperatorConfig()
  return normalizeAppSettings({
    ...settings.value,
    operator_config: operatorConfig
  })
}

const syncOperatorConfigText = () => {
  operatorConfigText.value = JSON.stringify(settings.value.operator_config, null, 2)
}

const syncSelectedModel = (preferredModel?: string | null) => {
  const optionNames = modelOptions.value.map((item) => item.name)
  if (preferredModel && optionNames.includes(preferredModel)) {
    settings.value.llm_model_default = preferredModel
    return
  }

  const defaultModel = modelCatalog.value?.default_model
  if (defaultModel && optionNames.includes(defaultModel)) {
    settings.value.llm_model_default = defaultModel
    return
  }

  settings.value.llm_model_default = optionNames[0] || defaultAppSettings.llm_model_default
}

const loadModelCatalog = async () => {
  try {
    modelCatalog.value = await getAvailableModels()
    syncSelectedModel(settings.value.llm_model_default)
  } catch (error) {
    ElMessage.error('加载模型列表失败')
  }
}

const formatModelLabel = (model: LLMModelOption) => {
  const status = model.installed ? '已安装' : '未安装'
  const defaultTag = model.is_default ? ' / 默认' : ''
  return `${model.name} (${model.strategy_label} / ${status}${defaultTag})`
}

const loadRuntimeInfo = async () => {
  try {
    engineInfo.value = await getEngineInfo()
  } catch (error) {
    ElMessage.error('加载运行时信息失败')
  }
}

const loadTemplateList = async () => {
  try {
    templates.value = await getTemplates()
  } catch (error) {
    ElMessage.error('加载模板列表失败')
  }
}

const saveSettings = async () => {
  try {
    settings.value = buildSettingsPayload()
    saveAppSettings(settings.value)
    ElMessage.success('设置已保存')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '保存设置失败')
  }
}

const resetSettings = () => {
  settings.value = normalizeAppSettings(defaultAppSettings)
  syncSelectedModel(settings.value.llm_model_default)
  syncOperatorConfigText()
  saveAppSettings(settings.value)
  ElMessage.success('已恢复默认设置')
}

const applyTemplate = async (template: ConfigTemplate, showMessage = true) => {
  settings.value = normalizeAppSettings({
    ...template.config_data,
    template_id: template.id,
    template_name: template.name
  })
  syncSelectedModel(settings.value.llm_model_default)
  syncOperatorConfigText()
  saveAppSettings(settings.value)

  if (showMessage) {
    ElMessage.success(`已应用模板：${template.name}`)
  }
}

const openTemplateDialog = () => {
  templateForm.value = {
    name: '',
    description: '',
    is_default: false
  }
  templateDialogVisible.value = true
}

const createTemplateFromCurrent = async () => {
  if (!templateForm.value.name.trim()) {
    ElMessage.warning('请输入模板名称')
    return
  }

  try {
    const currentSettings = buildSettingsPayload()
    const created = await createTemplate({
      name: templateForm.value.name.trim(),
      description: templateForm.value.description.trim(),
      is_default: templateForm.value.is_default,
      config_data: {
        ...currentSettings,
        template_id: null,
        template_name: null
      }
    })

    templateDialogVisible.value = false
    await loadTemplateList()
    await applyTemplate(created, false)
    ElMessage.success('模板已保存并应用')
  } catch (error) {
    ElMessage.error(error instanceof Error ? error.message : '创建模板失败')
  }
}

const showTemplatePreview = (template: ConfigTemplate) => {
  previewTemplate.value = template
  templatePreviewVisible.value = true
}

const removeTemplate = async (template: ConfigTemplate) => {
  try {
    await ElMessageBox.confirm(`确定删除模板“${template.name}”吗？`, '提示', {
      type: 'warning'
    })
    await deleteTemplate(template.id)

    if (settings.value.template_id === template.id) {
      settings.value = normalizeAppSettings({
        ...settings.value,
        template_id: null,
        template_name: null
      })
      saveAppSettings(settings.value)
    }

    await loadTemplateList()
    ElMessage.success('模板已删除')
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除模板失败')
    }
  }
}

const formatDate = (value?: string) => {
  if (!value) {
    return '-'
  }

  return new Date(value).toLocaleString('zh-CN')
}

onMounted(async () => {
  settings.value = loadAppSettings()
  syncOperatorConfigText()
  await Promise.all([loadRuntimeInfo(), loadTemplateList(), loadModelCatalog()])
})
</script>

<style scoped>
.settings {
  max-width: 1280px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.section-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
}

.section-subtitle {
  margin-top: 4px;
  font-size: 13px;
  color: #6b7280;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.template-name {
  display: flex;
  align-items: center;
  gap: 8px;
}

.template-chip {
  min-height: 32px;
  display: flex;
  align-items: center;
}

.full-width {
  width: 100%;
}

.tag-group {
  display: flex;
  flex-wrap: wrap;
}

.form-tip {
  margin-top: 8px;
  font-size: 12px;
  line-height: 1.6;
  color: #6b7280;
}

.json-preview {
  margin: 0;
  padding: 16px;
  border-radius: 8px;
  background: #f5f7fa;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: calc(100vh - 220px);
  overflow: auto;
}

.muted-text {
  color: #909399;
}

.mb-20 {
  margin-bottom: 20px;
}

.mb-8 {
  margin-bottom: 8px;
}

.mr-8 {
  margin-right: 8px;
}

@media (max-width: 768px) {
  .section-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .header-actions {
    width: 100%;
    flex-wrap: wrap;
  }
}
</style>
