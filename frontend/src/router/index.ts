import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      component: () => import('@/views/Layout.vue'),
      redirect: '/desensitize',
      children: [
        {
          path: 'setup',
          name: 'SetupGuide',
          component: () => import('@/views/SetupGuide.vue'),
          meta: { title: '启动检查' }
        },
        {
          path: 'desensitize',
          name: 'Desensitize',
          component: () => import('@/views/Desensitize.vue'),
          meta: { title: '文档脱敏' }
        },
        {
          path: 'custom-rules',
          name: 'CustomRules',
          component: () => import('@/views/CustomRules.vue'),
          meta: { title: '自定义规则' }
        },
        {
          path: 'settings',
          name: 'Settings',
          component: () => import('@/views/Settings.vue'),
          meta: { title: '系统设置' }
        }
      ]
    }
  ]
})

export default router
