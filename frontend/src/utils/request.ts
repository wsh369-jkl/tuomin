import axios from 'axios'
import { ElMessage } from 'element-plus'

const LONG_RUNNING_TIMEOUT_MS = 60 * 60 * 1000

const request = axios.create({
  baseURL: '/api/v1',
  timeout: LONG_RUNNING_TIMEOUT_MS
})

request.interceptors.request.use(
  config => config,
  error => Promise.reject(error)
)

request.interceptors.response.use(
  response => response.data,
  error => {
    const isTimeout =
      error.code === 'ECONNABORTED' || String(error.message || '').toLowerCase().includes('timeout')
    const message = isTimeout
      ? '处理时间超过当前等待上限。系统默认使用本地 4B 稳定增强策略，长文档在弱设备上会更慢，但会优先保证运行稳定。请稍后重试，或先缩小文档范围。'
      : error.response?.data?.detail || error.message || 'Request failed'

    ElMessage.error(message)
    return Promise.reject(error)
  }
)

export default request
