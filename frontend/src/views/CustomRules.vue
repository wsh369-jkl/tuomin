<template>
  <div class="custom-rules">
    <el-row :gutter="20" class="mb-20">
      <el-col :xs="24" :sm="8">
        <el-card>
          <el-statistic title="关键词规则组" :value="customConfig?.keywords_count ?? 0" />
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="8">
        <el-card>
          <el-statistic title="正则规则数" :value="customConfig?.patterns_count ?? 0" />
        </el-card>
      </el-col>
      <el-col :xs="24" :sm="8">
        <el-card>
          <el-statistic title="可识别实体类型" :value="keywordRows.length + patternRows.length" />
        </el-card>
      </el-col>
    </el-row>

    <el-card class="mb-20">
      <template #header>
        <div class="card-header">
          <div>
            <div class="card-title">自定义规则管理</div>
            <div class="card-subtitle">维护新引擎使用的关键词和正则规则</div>
          </div>
          <div class="header-actions">
            <el-button @click="handleReload">
              <el-icon><Refresh /></el-icon>
              重新加载
            </el-button>
            <el-button type="primary" @click="openCreateDialog(activeTab)">
              <el-icon><Plus /></el-icon>
              新增{{ activeTab === 'keywords' ? '关键词' : '正则' }}规则
            </el-button>
          </div>
        </div>
      </template>

      <el-tabs v-model="activeTab">
        <el-tab-pane label="关键词规则" name="keywords">
          <el-table :data="keywordRows" empty-text="暂无关键词规则">
            <el-table-column prop="entity_type" label="实体类型" min-width="160" />
            <el-table-column prop="description" label="说明" min-width="220" />
            <el-table-column label="关键词" min-width="280">
              <template #default="{ row }">
                <div class="tag-wrap">
                  <el-tag
                    v-for="keyword in row.keywords"
                    :key="keyword"
                    class="mr-8 mb-8"
                  >
                    {{ keyword }}
                  </el-tag>
                </div>
              </template>
            </el-table-column>
            <el-table-column label="置信度" width="110">
              <template #default="{ row }">
                {{ formatScore(row.score) }}
              </template>
            </el-table-column>
            <el-table-column label="操作" width="160">
              <template #default="{ row }">
                <el-button type="primary" link @click="openEditKeyword(row)">
                  编辑
                </el-button>
                <el-button type="danger" link @click="removeKeywordRule(row.entity_type)">
                  删除
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>

        <el-tab-pane label="正则规则" name="patterns">
          <el-table :data="patternRows" empty-text="暂无正则规则">
            <el-table-column prop="entity_type" label="实体类型" min-width="160" />
            <el-table-column prop="description" label="说明" min-width="180" />
            <el-table-column prop="regex" label="正则表达式" min-width="240" />
            <el-table-column label="上下文" min-width="200">
              <template #default="{ row }">
                <div class="tag-wrap" v-if="row.context.length">
                  <el-tag
                    v-for="item in row.context"
                    :key="item"
                    type="info"
                    class="mr-8 mb-8"
                  >
                    {{ item }}
                  </el-tag>
                </div>
                <span v-else class="muted-text">未设置</span>
              </template>
            </el-table-column>
            <el-table-column label="置信度" width="110">
              <template #default="{ row }">
                {{ formatScore(row.score) }}
              </template>
            </el-table-column>
            <el-table-column label="操作" width="160">
              <template #default="{ row }">
                <el-button type="primary" link @click="openEditPattern(row)">
                  编辑
                </el-button>
                <el-button type="danger" link @click="removePatternRule(row.entity_type)">
                  删除
                </el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-tab-pane>
      </el-tabs>
    </el-card>

    <el-card>
      <template #header>
        <div class="card-header">
          <div>
            <div class="card-title">规则效果测试</div>
            <div class="card-subtitle">输入一段文本，快速验证当前规则是否能命中</div>
          </div>
          <el-button type="primary" :disabled="!testText.trim()" @click="runTest">
            <el-icon><Search /></el-icon>
            开始测试
          </el-button>
        </div>
      </template>

      <el-input
        v-model="testText"
        type="textarea"
        :rows="5"
        placeholder="请输入要测试的合同或样本文本"
      />

      <div class="mt-20">
        <el-empty
          v-if="!testResult"
          description="测试结果会显示在这里"
        />

        <div v-else>
          <el-alert
            :title="`共识别到 ${testResult.count} 个实体`"
            type="success"
            :closable="false"
            class="mb-20"
          />
          <el-table :data="testResult.entities" empty-text="当前文本未命中自定义规则">
            <el-table-column prop="type" label="实体类型" width="180" />
            <el-table-column prop="text" label="命中文本" min-width="240" />
            <el-table-column label="来源" width="120">
              <template #default="{ row }">
                <el-tag :type="row.source === 'custom' ? 'warning' : 'info'">
                  {{ row.source }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column label="置信度" width="110">
              <template #default="{ row }">
                {{ formatScore(row.score) }}
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </el-card>

    <el-dialog
      v-model="dialogVisible"
      :title="dialogTitle"
      width="680px"
      destroy-on-close
    >
      <el-form label-width="110px">
        <el-form-item label="规则类型">
          <el-radio-group v-model="dialogRuleType" :disabled="isEdit">
            <el-radio-button label="keywords">关键词规则</el-radio-button>
            <el-radio-button label="patterns">正则规则</el-radio-button>
          </el-radio-group>
        </el-form-item>

        <el-form-item label="实体类型">
          <el-input
            v-model="form.entity_type"
            placeholder="例如 COMPANY_NAME / PROJECT_CODE"
            :disabled="isEdit"
          />
        </el-form-item>

        <el-form-item label="规则说明">
          <el-input
            v-model="form.description"
            placeholder="可选，用于说明该规则的用途"
          />
        </el-form-item>

        <template v-if="dialogRuleType === 'keywords'">
          <el-form-item label="关键词列表">
            <el-select
              v-model="form.keywords"
              multiple
              filterable
              allow-create
              default-first-option
              placeholder="输入关键词后按回车添加"
              style="width: 100%"
            />
          </el-form-item>
        </template>

        <template v-else>
          <el-form-item label="正则表达式">
            <el-input
              v-model="form.regex"
              placeholder="例如 HT-\\d{4}-\\d{3}"
            />
          </el-form-item>
          <el-form-item label="上下文提示">
            <el-select
              v-model="form.context"
              multiple
              filterable
              allow-create
              default-first-option
              placeholder="可选，用于提供上下文关键词"
              style="width: 100%"
            />
          </el-form-item>
        </template>

        <el-form-item label="置信度">
          <el-slider
            v-model="form.score"
            :min="0.5"
            :max="1"
            :step="0.05"
            :format-tooltip="formatScore"
          />
        </el-form-item>
      </el-form>

      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="saveRule">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Refresh, Search } from '@element-plus/icons-vue'
import {
  addKeywords,
  addPattern,
  deleteKeywords,
  deletePattern,
  getCustomConfig,
  reloadConfig,
  testRecognizer
} from '@/api/custom'
import type {
  CustomConfig,
  PatternRule,
  TestRecognizerResponse
} from '@/api/custom'

type RuleTab = 'keywords' | 'patterns'

interface KeywordRow {
  entity_type: string
  description: string
  keywords: string[]
  score: number
}

interface PatternRow {
  entity_type: string
  description: string
  regex: string
  context: string[]
  score: number
}

interface RuleFormState {
  entity_type: string
  description: string
  keywords: string[]
  regex: string
  context: string[]
  score: number
}

const activeTab = ref<RuleTab>('keywords')
const customConfig = ref<CustomConfig | null>(null)
const dialogVisible = ref(false)
const dialogRuleType = ref<RuleTab>('keywords')
const isEdit = ref(false)
const testText = ref('')
const testResult = ref<TestRecognizerResponse | null>(null)

const form = reactive<RuleFormState>({
  entity_type: '',
  description: '',
  keywords: [],
  regex: '',
  context: [],
  score: 0.95
})

const keywordRows = computed<KeywordRow[]>(() => {
  if (!customConfig.value) {
    return []
  }

  return Object.entries(customConfig.value.keywords).map(([entity_type, rule]) => ({
    entity_type,
    description: rule.description || '',
    keywords: [...rule.keywords],
    score: rule.score
  }))
})

const patternRows = computed<PatternRow[]>(() => {
  return (customConfig.value?.patterns ?? []).map((rule: PatternRule) => ({
    entity_type: rule.name,
    description: rule.description || '',
    regex: rule.regex,
    context: [...(rule.context ?? [])],
    score: rule.score
  }))
})

const dialogTitle = computed(() => {
  const action = isEdit.value ? '编辑' : '新增'
  return `${action}${dialogRuleType.value === 'keywords' ? '关键词规则' : '正则规则'}`
})

const formatScore = (value: number) => `${Math.round(value * 100)}%`

const normalizeList = (values: string[]) => {
  const next: string[] = []
  const seen = new Set<string>()

  values.forEach((value) => {
    const item = value.trim()
    if (!item || seen.has(item)) {
      return
    }
    next.push(item)
    seen.add(item)
  })

  return next
}

const resetForm = (ruleType: RuleTab) => {
  dialogRuleType.value = ruleType
  form.entity_type = ''
  form.description = ''
  form.keywords = []
  form.regex = ''
  form.context = []
  form.score = ruleType === 'keywords' ? 0.95 : 0.9
}

const loadConfig = async () => {
  customConfig.value = await getCustomConfig()
}

const openCreateDialog = (ruleType: RuleTab) => {
  isEdit.value = false
  resetForm(ruleType)
  dialogVisible.value = true
}

const openEditKeyword = (row: KeywordRow) => {
  isEdit.value = true
  dialogRuleType.value = 'keywords'
  form.entity_type = row.entity_type
  form.description = row.description
  form.keywords = [...row.keywords]
  form.regex = ''
  form.context = []
  form.score = row.score
  dialogVisible.value = true
}

const openEditPattern = (row: PatternRow) => {
  isEdit.value = true
  dialogRuleType.value = 'patterns'
  form.entity_type = row.entity_type
  form.description = row.description
  form.keywords = []
  form.regex = row.regex
  form.context = [...row.context]
  form.score = row.score
  dialogVisible.value = true
}

const saveRule = async () => {
  const entityType = form.entity_type.trim()
  if (!entityType) {
    ElMessage.warning('请先填写实体类型')
    return
  }

  if (dialogRuleType.value === 'keywords') {
    const keywords = normalizeList(form.keywords)
    if (!keywords.length) {
      ElMessage.warning('请至少添加一个关键词')
      return
    }

    await addKeywords({
      entity_type: entityType,
      description: form.description.trim(),
      keywords,
      score: form.score
    })
  } else {
    const regex = form.regex.trim()
    if (!regex) {
      ElMessage.warning('请填写正则表达式')
      return
    }

    await addPattern({
      entity_type: entityType,
      description: form.description.trim(),
      regex,
      context: normalizeList(form.context),
      score: form.score
    })
  }

  ElMessage.success('规则已保存')
  dialogVisible.value = false
  await loadConfig()
}

const removeKeywordRule = async (entityType: string) => {
  try {
    await ElMessageBox.confirm(`确定删除 ${entityType} 的关键词规则吗？`, '提示', {
      type: 'warning'
    })
    await deleteKeywords(entityType)
    ElMessage.success('关键词规则已删除')
    await loadConfig()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除关键词规则失败')
    }
  }
}

const removePatternRule = async (entityType: string) => {
  try {
    await ElMessageBox.confirm(`确定删除 ${entityType} 的正则规则吗？`, '提示', {
      type: 'warning'
    })
    await deletePattern(entityType)
    ElMessage.success('正则规则已删除')
    await loadConfig()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除正则规则失败')
    }
  }
}

const handleReload = async () => {
  await reloadConfig()
  await loadConfig()
  ElMessage.success('配置已重新加载')
}

const runTest = async () => {
  if (!testText.value.trim()) {
    ElMessage.warning('请输入测试文本')
    return
  }

  testResult.value = await testRecognizer(testText.value)
}

onMounted(async () => {
  try {
    await loadConfig()
  } catch (error) {
    ElMessage.error('加载规则配置失败')
  }
})
</script>

<style scoped>
.custom-rules {
  max-width: 1400px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

.card-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
}

.card-subtitle {
  margin-top: 4px;
  font-size: 13px;
  color: #6b7280;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.tag-wrap {
  display: flex;
  flex-wrap: wrap;
}

.muted-text {
  color: #909399;
}

.mb-20 {
  margin-bottom: 20px;
}

.mt-20 {
  margin-top: 20px;
}

.mb-8 {
  margin-bottom: 8px;
}

.mr-8 {
  margin-right: 8px;
}

@media (max-width: 768px) {
  .card-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .header-actions {
    width: 100%;
    flex-wrap: wrap;
  }
}
</style>
