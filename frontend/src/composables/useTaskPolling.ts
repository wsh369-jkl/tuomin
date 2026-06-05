import { onBeforeUnmount } from 'vue'

export const useTaskPolling = <TStatus, TResult>(options: {
  fetchStatus: (taskId: string) => Promise<TStatus>
  fetchResult: (taskId: string) => Promise<TResult>
  getStatus: (status: TStatus) => string
  isReady: (status: TStatus) => boolean
  isFailed?: (status: TStatus) => boolean
  intervalMs?: number
  onStatus?: (status: TStatus) => void
  onResult?: (result: TResult, status: TStatus) => void
  onFailed?: (status: TStatus) => void
  onError?: (error: unknown) => void
}) => {
  let timer: number | null = null

  const stop = () => {
    if (timer !== null) {
      window.clearTimeout(timer)
      timer = null
    }
  }

  const poll = async (taskId: string) => {
    try {
      const status = await options.fetchStatus(taskId)
      options.onStatus?.(status)

      if ((options.isFailed && options.isFailed(status)) || options.getStatus(status) === 'failed') {
        stop()
        options.onFailed?.(status)
        return
      }

      if (options.isReady(status)) {
        const result = await options.fetchResult(taskId)
        stop()
        options.onResult?.(result, status)
        return
      }

      timer = window.setTimeout(() => {
        void poll(taskId)
      }, options.intervalMs ?? 1200)
    } catch (error) {
      stop()
      options.onError?.(error)
    }
  }

  onBeforeUnmount(() => {
    stop()
  })

  return {
    poll,
    stop
  }
}
