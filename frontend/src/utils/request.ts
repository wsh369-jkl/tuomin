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
      ? '处理时间超过当前等待上限。系统默认使用本地高质量低内存流程，但长文档、扫描件或复杂文书仍可能较慢。请稍后重试，或先缩小文档范围。'
      : error.response?.data?.detail || error.message || 'Request failed'

    ElMessage.error(message)
    return Promise.reject(error)
  }
)

export default request
