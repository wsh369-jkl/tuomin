<template>
  <div class="layout">
    <el-container>
      <el-aside width="220px" class="layout-aside">
        <div class="logo">
          <h2>合同脱敏系统</h2>
          <p>Stable 4B Runtime</p>
        </div>

        <el-menu
          :default-active="activeMenu"
          router
          background-color="#0f172a"
          text-color="#cbd5e1"
          active-text-color="#f8fafc"
        >
          <el-menu-item index="/setup">
            <el-icon><Monitor /></el-icon>
            <span>启动检查</span>
          </el-menu-item>
          <el-menu-item index="/desensitize">
            <el-icon><Document /></el-icon>
            <span>文档脱敏</span>
          </el-menu-item>
          <el-menu-item index="/custom-rules">
            <el-icon><SetUp /></el-icon>
            <span>自定义规则</span>
          </el-menu-item>
          <el-menu-item index="/settings">
            <el-icon><Tools /></el-icon>
            <span>系统设置</span>
          </el-menu-item>
        </el-menu>
      </el-aside>

      <el-container>
        <el-header class="layout-header">
          <div class="header-content">
            <div>
              <h3>{{ currentTitle }}</h3>
              <p>当前界面已围绕 4B 稳定运行、跨平台联通和高质量脱敏流程进行收敛。</p>
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
              <el-button link type="primary" @click="goToSetup">去完成启动检查</el-button>
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
import { Document, Monitor, SetUp, Tools } from '@element-plus/icons-vue'
import { getRuntimeStatus, type RuntimeStatusResponse } from '@/api/desensitize'

const route = useRoute()
const router = useRouter()
const runtimeStatus = ref<RuntimeStatusResponse | null>(null)

const activeMenu = computed(() => route.path)
const currentTitle = computed(() => (route.meta.title as string) || '')

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
