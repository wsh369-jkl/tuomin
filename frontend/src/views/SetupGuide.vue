<template>
  <div class="setup-page">
    <div class="hero-card">
      <div class="hero-copy">
        <div class="eyebrow">First Launch Check</div>
        <h1>启动检查与初始化</h1>
        <p>{{ modeIntro }}</p>
        <div class="hero-actions">
          <el-button type="primary" :loading="loading" @click="refreshStatus">重新检测</el-button>
          <el-button @click="goToWorkspace" :disabled="!runtimeStatus?.ready">进入工作台</el-button>
        </div>
      </div>
      <div class="hero-status">
        <div class="status-shell" :class="runtimeStatus?.ready ? 'is-ready' : 'is-pending'">
          <div class="status-label">当前状态</div>
          <div class="status-value">
            {{ runtimeStatus?.ready ? '已就绪' : '待完成初始化' }}
          </div>
          <div class="status-hint">
            {{ runtimeStatus?.recommended_action || '正在读取本地运行状态...' }}
          </div>
        </div>
      </div>
    </div>

    <el-row :gutter="20">
      <el-col :xs="24" :xl="14">
        <el-card class="panel-card">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">运行状态</div>
                <div class="panel-subtitle">{{ runtimePanelSubtitle }}</div>
              </div>
            </div>
          </template>

          <el-skeleton :loading="loading" animated :rows="6">
            <template #default>
              <el-descriptions v-if="runtimeStatus" :column="2" border class="mb-16">
                <el-descriptions-item label="平台">
                  {{ platformLabel }}
                </el-descriptions-item>
                <el-descriptions-item label="运行后端">
                  {{ runtimeStatus.backend }}
                </el-descriptions-item>
                <el-descriptions-item :label="isHighQualityLowmem ? '按需精审模型' : '默认模型'">
                  {{ runtimeStatus.required_model || '-' }}
                </el-descriptions-item>
                <el-descriptions-item label="当前可用模型">
                  {{ runtimeStatus.preferred_processing_model || '-' }}
                </el-descriptions-item>
                <el-descriptions-item label="安装入口">
                  {{ runtimeStatus.installer_hint }}
                </el-descriptions-item>
                <el-descriptions-item v-if="!isHighQualityLowmem" label="Ollama 安装检测">
                  <el-tag :type="runtimeStatus.ollama_install_detected ? 'success' : 'danger'">
                    {{ runtimeStatus.ollama_install_detected ? '已检测到' : '未检测到' }}
                  </el-tag>
                </el-descriptions-item>
                <el-descriptions-item v-if="!isHighQualityLowmem" label="Ollama 服务">
                  <el-tag :type="runtimeStatus.service_available ? 'success' : 'warning'">
                    {{ runtimeStatus.service_available ? '可连接' : '未连接' }}
                  </el-tag>
                </el-descriptions-item>
                <el-descriptions-item
                  v-if="isHighQualityLowmem"
                  label="中文主识别模型"
                >
                  <el-tag :type="runtimeStatus.primary_models_ready ? 'success' : 'warning'">
                    {{ runtimeStatus.primary_models_ready ? '已就绪' : '未就绪' }}
                  </el-tag>
                </el-descriptions-item>
                <el-descriptions-item :label="isHighQualityLowmem ? '按需精审模型' : '默认 4B 模型'">
                  <el-tag :type="runtimeStatus.required_model_installed ? 'success' : 'warning'">
                    {{ runtimeStatus.required_model_installed ? '已安装' : '未安装' }}
                  </el-tag>
                </el-descriptions-item>
                <el-descriptions-item label="整体状态">
                  <el-tag :type="runtimeStatus.ready ? 'success' : 'danger'" size="large">
                    {{ runtimeStatus.ready ? '可以进入正式处理' : '仍需初始化' }}
                  </el-tag>
                </el-descriptions-item>
              </el-descriptions>

              <el-alert
                v-if="runtimeStatus"
                class="mb-16"
                :title="runtimeStatus.recommended_action"
                :type="runtimeStatus.ready ? 'success' : 'warning'"
                :closable="false"
              />

              <el-alert
                v-if="!isHighQualityLowmem && runtimeStatus?.ollama_path"
                class="mb-16"
                type="info"
                :closable="false"
                :title="`检测到本机 Ollama 路径：${runtimeStatus.ollama_path}`"
              />

              <el-empty
                v-if="!runtimeStatus"
                description="当前未获取到运行状态，请稍后重新检测。"
              />
            </template>
          </el-skeleton>
        </el-card>
      </el-col>

      <el-col :xs="24" :xl="10">
        <el-card class="panel-card">
          <template #header>
            <div class="panel-header">
              <div>
                <div class="panel-title">操作步骤</div>
              <div class="panel-subtitle">尽量按顺序完成，确保默认识别与脱敏主流程稳定可用。</div>
              </div>
            </div>
          </template>

          <div class="step-list">
            <div v-for="item in steps" :key="item.title" class="step-card">
              <div class="step-index">{{ item.index }}</div>
              <div class="step-body">
                <div class="step-title">{{ item.title }}</div>
                <div class="step-description">{{ item.description }}</div>
              </div>
            </div>
          </div>

          <el-divider />

          <div class="command-panel">
            <div class="command-title">建议命令或脚本</div>
            <div class="command-description">
              {{ runtimeStatus?.download_hint || '可先运行模型下载脚本，或手动执行 ollama pull。' }}
            </div>
            <pre class="command-block">{{ commandHint }}</pre>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { getRuntimeStatus, type RuntimeStatusResponse } from '@/api/desensitize'

const router = useRouter()
const loading = ref(false)
const runtimeStatus = ref<RuntimeStatusResponse | null>(null)

const isHighQualityLowmem = computed(
  () => runtimeStatus.value?.desensitize_mode === 'high_quality_lowmem'
)

const modeIntro = computed(() =>
  isHighQualityLowmem.value
    ? '这一页用于确认本机是否已经满足高质量低内存中文脱敏要求。默认路线是外挂规则、合同结构规则、中文实体模型主识别和小 Qwen 按需补漏，模型全部本地运行。'
    : '这一页用于确认本机是否已经满足文本脱敏处理要求。只有运行环境就绪后，系统才会开放正式处理入口。'
)

const runtimePanelSubtitle = computed(() =>
  isHighQualityLowmem.value
    ? '用于确认中文主识别模型、按需精审模型和当前客户端入口是否就绪。'
    : '用于确认 Ollama、可用处理模型和当前客户端入口是否就绪。'
)

const platformLabel = computed(() => {
  if (!runtimeStatus.value) {
    return '-'
  }

  const labelMap: Record<string, string> = {
    windows: 'Windows',
    macos: 'macOS',
    linux: 'Linux'
  }

  return labelMap[runtimeStatus.value.platform] || runtimeStatus.value.platform
})

const steps = computed(() => {
  const modelName = runtimeStatus.value?.required_model || 'qwen3.5:4b'
  const preferredModel = runtimeStatus.value?.preferred_processing_model || ''

  if (isHighQualityLowmem.value) {
    return [
      {
        index: '01',
        title: '确认中文主识别模型',
        description: runtimeStatus.value?.primary_models_ready
          ? '中文实体模型与中文 NER 主识别模型已就绪，外挂规则、合同结构回填和主召回可以直接运行。'
          : '请先下载中文实体模型与中文 NER 主识别模型；它们负责稳定召回，不依赖 Ollama 4B 全文抽取。'
      },
      {
        index: '02',
        title: `确认按需精审模型 ${modelName}`,
        description: runtimeStatus.value?.review_model_installed
          ? `当前已检测到按需精审模型，系统只在缺口片段加载 ${preferredModel || modelName}，处理结束后释放。`
          : '精审模型未安装时主流程仍可运行，但会标记人工复核，不伪装成完整高质量结果。'
      },
      {
        index: '03',
        title: '重新检测并进入正式处理',
        description: runtimeStatus.value?.ready
          ? '当前环境已经满足高质量低内存脱敏要求，可以直接进入文档脱敏。'
          : '完成模型下载后点击“重新检测”，状态通过后再进入文档脱敏。'
      }
    ]
  }

  return [
    {
      index: '01',
      title: '确认 Ollama 已安装',
      description: runtimeStatus.value?.ollama_install_detected
        ? '当前已经检测到本机 Ollama 安装。后续如仍显示服务未连接，请先手动打开 Ollama。'
        : '请先安装 Ollama。传统客户端分发形态下，模型仍保持本地运行，不会改成远端调用。'
    },
    {
      index: '02',
      title: `安装处理模型 ${modelName}`,
      description: runtimeStatus.value?.ready && preferredModel
        ? `当前已检测到可用处理模型 ${preferredModel}。如需默认 4B 稳态路线，可额外补装 ${modelName}。`
        : runtimeStatus.value?.required_model_installed
        ? '当前已经检测到默认 4B 模型。普通识别与脱敏会优先走这条稳态路线。'
        : `请先下载默认模型 ${modelName}，或安装 qwen3.5:27b 系列模型。当前系统不会开放正式处理入口，以避免环境不完整影响质量。`
    },
    {
      index: '03',
      title: '重新检测并进入正式处理',
      description: runtimeStatus.value?.ready
        ? '当前环境已经满足要求，可以直接进入文档脱敏。'
        : '完成前两步后点击“重新检测”，状态通过后再进入文档脱敏。'
    }
  ]
})

const commandHint = computed(() => {
  const modelName = runtimeStatus.value?.required_model || 'qwen3.5:4b'
  const platform = runtimeStatus.value?.platform || ''

  if (isHighQualityLowmem.value) {
    return [
      'python3 backend/bin/download_lowmem_models.py',
      runtimeStatus.value?.download_hint || '模型目录：-',
      '中文实体模型：uer/roberta-base-finetuned-cluener2020-chinese',
      '中文 NER：p988744/eland-ner-zh',
      '按需精审：Qwen/Qwen3-1.7B-MLX-4bit',
      '兜底：unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_K_M.gguf'
    ].join('\n')
  }

  if (platform === 'windows') {
    return `download_ollama_model.bat\nollama pull ${modelName}`
  }
  if (platform === 'macos') {
    return `./download_ollama_model.command\nollama pull ${modelName}`
  }
  return `ollama pull ${modelName}`
})

const refreshStatus = async () => {
  loading.value = true
  try {
    runtimeStatus.value = await getRuntimeStatus()
  } catch (error) {
    ElMessage.error('读取运行环境状态失败')
  } finally {
    loading.value = false
  }
}

const goToWorkspace = () => {
  if (!runtimeStatus.value?.ready) {
    ElMessage.warning('请先完成运行环境初始化，再进入正式处理。')
    return
  }
  router.push('/workspace')
}

onMounted(async () => {
  await refreshStatus()
})
</script>

<style scoped>
.setup-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
  max-width: 1500px;
}

.hero-card {
  display: grid;
  grid-template-columns: 1.45fr 0.85fr;
  gap: 20px;
  padding: 28px;
  border-radius: 24px;
  background:
    radial-gradient(circle at top left, rgba(22, 163, 74, 0.22), transparent 28%),
    radial-gradient(circle at bottom right, rgba(14, 165, 233, 0.2), transparent 32%),
    linear-gradient(135deg, #0f172a 0%, #111827 55%, #0b1120 100%);
  color: #f8fafc;
}

.eyebrow {
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: rgba(226, 232, 240, 0.72);
}

.hero-copy h1 {
  margin: 10px 0 0;
  font-size: 32px;
  line-height: 1.15;
}

.hero-copy p {
  margin: 14px 0 0;
  max-width: 760px;
  line-height: 1.75;
  color: rgba(226, 232, 240, 0.82);
}

.hero-actions {
  display: flex;
  gap: 12px;
  margin-top: 22px;
}

.hero-status {
  display: flex;
  align-items: center;
}

.status-shell {
  width: 100%;
  padding: 22px;
  border-radius: 20px;
  background: rgba(15, 23, 42, 0.54);
  border: 1px solid rgba(148, 163, 184, 0.22);
}

.status-shell.is-ready {
  box-shadow: inset 0 0 0 1px rgba(34, 197, 94, 0.2);
}

.status-shell.is-pending {
  box-shadow: inset 0 0 0 1px rgba(251, 191, 36, 0.2);
}

.status-label {
  font-size: 13px;
  color: rgba(226, 232, 240, 0.72);
}

.status-value {
  margin-top: 8px;
  font-size: 26px;
  font-weight: 700;
}

.status-hint {
  margin-top: 10px;
  line-height: 1.7;
  color: rgba(226, 232, 240, 0.86);
}

.panel-card {
  min-height: 480px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.panel-title {
  font-size: 16px;
  font-weight: 600;
  color: #0f172a;
}

.panel-subtitle {
  margin-top: 4px;
  font-size: 13px;
  color: #64748b;
}

.step-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.step-card {
  display: flex;
  gap: 14px;
  padding: 16px;
  border-radius: 16px;
  background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
  border: 1px solid rgba(148, 163, 184, 0.18);
}

.step-index {
  width: 44px;
  height: 44px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #0f172a;
  color: #f8fafc;
  font-size: 14px;
  font-weight: 700;
}

.step-title {
  font-size: 15px;
  font-weight: 600;
  color: #0f172a;
}

.step-description {
  margin-top: 6px;
  font-size: 13px;
  line-height: 1.7;
  color: #475569;
}

.command-panel {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.command-title {
  font-size: 14px;
  font-weight: 600;
  color: #0f172a;
}

.command-description {
  font-size: 13px;
  line-height: 1.7;
  color: #475569;
}

.command-block {
  margin: 0;
  padding: 16px;
  border-radius: 14px;
  background: #0f172a;
  color: #e2e8f0;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: 'Consolas', 'Courier New', monospace;
}

.mb-16 {
  margin-bottom: 16px;
}

@media (max-width: 1024px) {
  .hero-card {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .hero-card {
    padding: 20px;
  }

  .hero-copy h1 {
    font-size: 26px;
  }

  .hero-actions {
    flex-direction: column;
  }
}
</style>
