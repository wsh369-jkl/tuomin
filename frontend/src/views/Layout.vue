<template>
  <div class="layout">
    <el-container>
      <el-aside width="220px" class="layout-aside">
        <div class="logo">
          <h2>{{ highQualityOnly ? '合同脱敏系统' : '本地文档工作台' }}</h2>
          <p>{{ highQualityOnly ? 'High Quality Low Memory' : 'Text Desensitize / PDF Audit' }}</p>
        </div>

        <el-menu
          :default-active="activeMenu"
          router
          background-color="#0f172a"
          text-color="#cbd5e1"
          active-text-color="#f8fafc"
        >
          <template v-if="highQualityOnly">
            <el-menu-item index="/desensitize">
              <el-icon><Document /></el-icon>
              <span>高质量脱敏</span>
            </el-menu-item>
          </template>
          <template v-else>
            <el-menu-item index="/workspace">
              <el-icon><Grid /></el-icon>
              <span>工作台</span>
            </el-menu-item>
            <el-menu-item-group title="功能分区">
              <el-menu-item index="/desensitize">
                <el-icon><Document /></el-icon>
                <span>文本脱敏</span>
              </el-menu-item>
              <el-menu-item index="/pdf-word-audit">
                <el-icon><Tickets /></el-icon>
                <span>转 Word 核查</span>
              </el-menu-item>
            </el-menu-item-group>
            <el-menu-item-group title="系统">
              <el-menu-item index="/setup">
                <el-icon><Monitor /></el-icon>
                <span>启动检查</span>
              </el-menu-item>
              <el-menu-item index="/custom-rules">
                <el-icon><SetUp /></el-icon>
                <span>自定义规则</span>
              </el-menu-item>
              <el-menu-item index="/settings">
                <el-icon><Tools /></el-icon>
                <span>系统设置</span>
              </el-menu-item>
            </el-menu-item-group>
          </template>
        </el-menu>
      </el-aside>

      <el-container>
        <el-header class="layout-header">
          <div class="header-content">
            <div>
              <h3>{{ currentTitle }}</h3>
              <p>{{ currentDescription }}</p>
            </div>
          </div>
        </el-header>
        <el-main class="layout-main">
          <el-alert
            v-if="runtimeStatus && !runtimeStatus.ready"
            class="runtime-banner"
            type="warning"
            :closable="false"
            show-icon
          >
            <template #title>
              <div class="runtime-banner-title">运行环境尚未完全就绪</div>
            </template>
            <div class="runtime-banner-body">
              <span>{{ runtimeStatus.recommended_action }}</span>
              <el-button v-if="!highQualityOnly" link type="primary" @click="goToSetup">去完成启动检查</el-button>
            </div>
          </el-alert>
          <router-view />
        </el-main>
      </el-container>
    </el-container>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Document, Grid, Monitor, SetUp, Tickets, Tools } from '@element-plus/icons-vue'
import { getRuntimeStatus, type RuntimeStatusResponse } from '@/api/desensitize'

const route = useRoute()
const router = useRouter()
const runtimeStatus = ref<RuntimeStatusResponse | null>(null)
const highQualityOnly = import.meta.env.VITE_HIGH_QUALITY_ONLY === '1'

const activeMenu = computed(() => route.path)
const currentTitle = computed(() => (route.meta.title as string) || '')
const currentDescription = computed(
  () => (route.meta.description as string) || '当前页面正在使用本地文档工作台。'
)

const refreshRuntimeStatus = async () => {
  try {
    runtimeStatus.value = await getRuntimeStatus()
  } catch (error) {
    runtimeStatus.value = null
  }
}

const goToSetup = () => {
  router.push('/setup')
}

const handleWindowFocus = () => {
  void refreshRuntimeStatus()
}

watch(
  () => route.fullPath,
  () => {
    void refreshRuntimeStatus()
  }
)

onMounted(async () => {
  window.addEventListener('focus', handleWindowFocus)
  await refreshRuntimeStatus()
})

onUnmounted(() => {
  window.removeEventListener('focus', handleWindowFocus)
})
</script>

<style scoped>
.layout {
  width: 100%;
  height: 100vh;
  background:
    radial-gradient(circle at top left, rgba(14, 165, 233, 0.16), transparent 30%),
    linear-gradient(180deg, #eef4ff 0%, #f8fafc 100%);
}

.el-container {
  height: 100%;
}

.layout-aside {
  background: linear-gradient(180deg, #0f172a 0%, #111827 100%);
  border-right: 1px solid rgba(148, 163, 184, 0.18);
}

.logo {
  padding: 24px 20px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.14);
}

.logo h2 {
  margin: 0;
  font-size: 18px;
  color: #f8fafc;
}

.logo p {
  margin: 6px 0 0;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #94a3b8;
}

.el-menu {
  border-right: none;
}

.el-menu-item {
  margin: 6px 10px;
  border-radius: 10px;
}

.el-menu :deep(.el-menu-item-group__title) {
  padding: 14px 20px 6px;
  color: #64748b;
  font-size: 12px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.el-menu-item.is-active {
  background: linear-gradient(135deg, rgba(56, 189, 248, 0.28), rgba(37, 99, 235, 0.42));
}

.layout-header {
  display: flex;
  align-items: center;
  background: rgba(255, 255, 255, 0.72);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(148, 163, 184, 0.16);
  padding: 0 28px;
}

.header-content h3 {
  margin: 0;
  font-size: 20px;
  color: #0f172a;
}

.header-content p {
  margin: 6px 0 0;
  color: #64748b;
  font-size: 13px;
}

.layout-main {
  padding: 24px 28px;
}

.runtime-banner {
  margin-bottom: 18px;
}

.runtime-banner-title {
  font-weight: 600;
}

.runtime-banner-body {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
}

@media (max-width: 768px) {
  .layout-main {
    padding: 16px;
  }

  .runtime-banner-body {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
