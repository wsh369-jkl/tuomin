import { createRouter, createWebHistory } from 'vue-router'

const highQualityOnly = import.meta.env.VITE_HIGH_QUALITY_ONLY === '1'

const router = createRouter({
  history: createWebHistory(),
  routes: highQualityOnly
    ? [
        {
          path: '/',
          component: () => import('@/views/Layout.vue'),
          redirect: '/desensitize',
          children: [
            {
              path: 'desensitize',
              name: 'Desensitize',
              component: () => import('@/views/Desensitize.vue'),
              meta: {
                title: '高质量低内存脱敏',
                description: '仅保留本地高质量低内存识别、脱敏和导出流程。'
              }
            }
          ]
        },
        {
          path: '/:pathMatch(.*)*',
          redirect: '/desensitize'
        }
      ]
    : [
        {
          path: '/',
          component: () => import('@/views/Layout.vue'),
          redirect: '/workspace',
          children: [
            {
              path: 'workspace',
              name: 'Workspace',
              component: () => import('@/views/Workspace.vue'),
              meta: {
                title: '工作台',
                description: '从这里进入文本脱敏、律师协助或 PDF 转 Word 核查，三块功能分区独立运行。'
              }
            },
            {
              path: 'setup',
              name: 'SetupGuide',
              component: () => import('@/views/SetupGuide.vue'),
              meta: {
                title: '启动检查',
                description: '检查本机运行环境、Ollama 服务和默认模型是否可用。'
              }
            },
            {
              path: 'desensitize',
              name: 'Desensitize',
              component: () => import('@/views/Desensitize.vue'),
              meta: {
                title: '文本脱敏',
                description: '这里只处理文本识别、脱敏和导出，不承载律师协助功能。'
              }
            },
            {
              path: 'assistant',
              name: 'Assistant',
              component: () => import('@/views/Review.vue'),
              meta: {
                title: '律师协助',
                description: '独立的律师辅助工作流，优先面向诉讼、执行、保全和证据材料。'
              }
            },
            {
              path: 'pdf-word-audit',
              name: 'PdfWordAudit',
              component: () => import('@/views/PdfWordAudit.vue'),
              meta: {
                title: 'PDF 转 Word 核查',
                description: '上传原 PDF 和 WPS 转换 DOCX，核查 OCR 差异并输出带批注的 Word 文档和证据报告。'
              }
            },
            {
              path: 'review',
              redirect: (to) => ({
                path: '/assistant',
                query: to.query
              })
            },
            {
              path: 'custom-rules',
              name: 'CustomRules',
              component: () => import('@/views/CustomRules.vue'),
              meta: {
                title: '自定义规则',
                description: '维护识别与脱敏使用的自定义规则和模板。'
              }
            },
            {
              path: 'settings',
              name: 'Settings',
              component: () => import('@/views/Settings.vue'),
              meta: {
                title: '系统设置',
                description: '配置默认识别模型、隐名策略和运行时参数。'
              }
            }
          ]
        }
      ]
})

export default router
