import { onBeforeUnmount, onMounted } from 'vue'

const HEARTBEAT_INTERVAL_MS = 5000
const SESSION_BASE_PATH = '/api/v1/desensitize/session'

type SessionPayload = {
  page_session_id: string
}

const buildSessionId = () => {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }

  return `page-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

const postSessionPayload = async (path: string, payload: SessionPayload, keepalive = false) => {
  await fetch(`${SESSION_BASE_PATH}${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload),
    credentials: 'same-origin',
    keepalive
  })
}

export const usePageSession = () => {
  const pageSessionId = buildSessionId()
  let heartbeatTimer: number | null = null
  let closed = false

  const stopHeartbeat = () => {
    if (heartbeatTimer !== null) {
      window.clearInterval(heartbeatTimer)
      heartbeatTimer = null
    }
  }

  const heartbeat = async () => {
    if (closed) {
      return
    }

    try {
      await postSessionPayload('/heartbeat', { page_session_id: pageSessionId })
    } catch {
      // Ignore transient heartbeat failures. The backend watchdog handles timeouts.
    }
  }

  const closeSession = async () => {
    if (closed) {
      return
    }

    closed = true
    stopHeartbeat()
    try {
      await postSessionPayload('/close', { page_session_id: pageSessionId }, true)
    } catch {
      // Ignore close failures. The server-side heartbeat watchdog is the fallback.
    }
  }

  const sendCloseBeacon = () => {
    if (closed) {
      return
    }

    closed = true
    stopHeartbeat()
    const payload = JSON.stringify({ page_session_id: pageSessionId })

    if (navigator.sendBeacon) {
      const blob = new Blob([payload], { type: 'application/json' })
      navigator.sendBeacon(`${SESSION_BASE_PATH}/close`, blob)
      return
    }

    void fetch(`${SESSION_BASE_PATH}/close`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: payload,
      credentials: 'same-origin',
      keepalive: true
    })
  }

  const handlePageHide = () => {
    sendCloseBeacon()
  }

  onMounted(() => {
    void heartbeat()
    heartbeatTimer = window.setInterval(() => {
      void heartbeat()
    }, HEARTBEAT_INTERVAL_MS)
    window.addEventListener('pagehide', handlePageHide)
  })

  onBeforeUnmount(() => {
    window.removeEventListener('pagehide', handlePageHide)
    void closeSession()
  })

  return {
    pageSessionId,
    closeSession
  }
}
