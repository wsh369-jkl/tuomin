<template>
  <div class="workspace-page">
    <div class="hero-card">
      <div class="hero-copy">
        <div class="eyebrow">Workspace</div>
        <h1>本地文档工作台</h1>
        <p>
          当前系统分成两块独立功能区。文本脱敏负责识别、脱敏与导出；PDF 转 Word 核查负责复核
          WPS 转换底稿，两边互不干扰。
        </p>
      </div>
      <div class="hero-status">
        <div class="status-card" :class="runtimeStatus?.ready ? 'is-ready' : 'is-pending'">
          <div class="status-label">运行状态</div>
          <div class="status-value">{{ runtimeStatus?.ready ? '已就绪' : '待初始化' }}</div>
          <div class="status-hint">
            {{ runtimeStatus?.recommended_action || '正在读取运行环境...' }}
          </div>
        </div>
      </div>
    </div>

    <el-row :gutter="20" class="entry-grid">
      <el-col :xs="24" :lg="12">
        <el-card shadow="hover" class="entry-card">
          <div class="entry-eyebrow">独立功能区 A</div>
          <div class="entry-title">文本脱敏</div>
          <div class="entry-description">
            上传文档后执行文本识别、实体抽取、脱敏替换和文件导出。这个分区只处理脱敏流程。
          </div>
          <div class="entry-points">
            <el-tag effect="plain">识别</el-tag>
            <el-tag effect="plain">脱敏</el-tag>
            <el-tag effect="plain">导出</el-tag>
          </div>
          <el-button type="primary" size="large" @click="router.push('/desensitize')">
            进入文本脱敏
          </el-button>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="12">
        <el-card shadow="hover" class="entry-card">
          <div class="entry-eyebrow">独立功能区 B</div>
          <div class="entry-title">PDF 转 Word 核查</div>
          <div class="entry-description">
            使用 WPS 转换 DOCX 保留版面，再用本机 OCR 和审查模型复核文本差异，只写入批注和证据报告。
          </div>
          <div class="entry-points">
            <el-tag type="success" effect="plain">模型复核</el-tag>
            <el-tag type="warning" effect="plain">人工复核</el-tag>
            <el-tag type="info" effect="plain">WPS 批注</el-tag>
          </div>
          <el-button
            type="primary"
            plain
            size="large"
            :disabled="!runtimeStatus?.ready"
            @click="router.push('/pdf-word-audit')"
          >
            进入转 Word 核查
          </el-button>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { getRuntimeStatus, type RuntimeStatusResponse } from '@/api/desensitize'

const router = useRouter()
const runtimeStatus = ref<RuntimeStatusResponse | null>(null)

const refreshRuntimeStatus = async () => {
  try {
    runtimeStatus.value = await getRuntimeStatus()
  } catch {
    runtimeStatus.value = null
  }
}

onMounted(async () => {
  await refreshRuntimeStatus()
})
</script>

<style scoped>
.workspace-page {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.hero-card {
  display: grid;
  grid-template-columns: minmax(0, 1.7fr) minmax(280px, 0.9fr);
  gap: 20px;
  padding: 28px;
  border-radius: 20px;
  background:
    radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 28%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(241, 245, 249, 0.96));
  border: 1px solid rgba(148, 163, 184, 0.22);
}

.eyebrow,
.entry-eyebrow {
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #64748b;
}

.hero-copy h1 {
  margin: 10px 0 12px;
  font-size: 30px;
  color: #0f172a;
}

.hero-copy p,
.entry-description,
.status-hint {
  color: #475569;
  line-height: 1.8;
}

.status-card {
  height: 100%;
  border-radius: 18px;
  padding: 20px;
  background: #f8fafc;
  border: 1px solid #dbeafe;
}

.status-card.is-ready {
  background: linear-gradient(180deg, #ecfdf5 0%, #f8fafc 100%);
  border-color: #86efac;
}

.status-card.is-pending {
  background: linear-gradient(180deg, #fff7ed 0%, #f8fafc 100%);
  border-color: #fdba74;
}

.status-label {
  font-size: 13px;
  color: #64748b;
}

.status-value,
.entry-title {
  margin-top: 10px;
  font-size: 24px;
  font-weight: 700;
  color: #0f172a;
}

.workflow-selector {
  margin-top: 16px;
  padding-top: 14px;
  border-top: 1px solid rgba(148, 163, 184, 0.24);
}

.workflow-selector :deep(.el-radio-group) {
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
}

.workflow-hint {
  margin-top: 10px;
  color: #475569;
  font-size: 13px;
  line-height: 1.6;
}

.entry-grid {
  margin-top: 4px;
}

.entry-card {
  height: 100%;
  border-radius: 18px;
}

.entry-points {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 18px 0 24px;
}

.entry-card :deep(.el-button) {
  width: 100%;
}

@media (max-width: 992px) {
  .hero-card {
    grid-template-columns: 1fr;
  }
}
</style>
