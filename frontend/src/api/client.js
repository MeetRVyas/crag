import axios from 'axios'

// ── Axios instance ──────────────────────────────────────────────────────────
const BASE = import.meta.env.VITE_API_URL ?? '/api'

export const client = axios.create({
  baseURL: BASE,
  timeout: 60_000,
})

// Attach JWT from localStorage to every request
client.interceptors.request.use((config) => {
  const token = localStorage.getItem('crag_token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

// On 401 → clear token (session expired)
client.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('crag_token')
      window.dispatchEvent(new Event('crag:logout'))
    }
    return Promise.reject(err)
  },
)

// ── Auth ────────────────────────────────────────────────────────────────────
export const api = {
  auth: {
    /** Returns the URL to redirect the user to for Google OAuth */
    loginUrl: () => `${BASE}/auth/login`,

    /** Store provider API keys for the session */
    setKeys: (keys) => client.post('/auth/set_keys', { keys }),

    logout: () => client.post('/auth/logout'),
  },

  // ── Documents ─────────────────────────────────────────────────────────────
  documents: {
    list: () => client.get('/documents/'),

    upload: (file, onProgress) => {
      const fd = new FormData()
      fd.append('file', file)
      return client.post('/documents/upload', fd, {
        onUploadProgress: (e) => onProgress?.(Math.round((e.loaded / e.total) * 100)),
      })
    },

    process: (provider, embedding_model) =>
      client.post('/documents/process', { provider, embedding_model }),

    delete: (filename) => client.delete(`/documents/${encodeURIComponent(filename)}`),

    snapshots: () => client.get('/documents/snapshots'),

    query: (query, model, provider) =>
      client.post('/documents/query', { query, model, provider }),
  },

  // ── CRAG ──────────────────────────────────────────────────────────────────
  crag: {
    chat: (payload) => client.post('/crag/chat', payload),

    /** Returns an EventSource for SSE pipeline status */
    statusStream: (token) => {
      const url = `${BASE}/crag/status`
      return new EventSource(url, {
        withCredentials: false,
        // We can't set headers on EventSource — pass token in the request
        // via a cookie or query param if needed; for now rely on session
      })
    },

    cache: (snapshot_ids) => {
      const params = snapshot_ids?.length
        ? snapshot_ids.map((id) => `snapshot_ids=${id}`).join('&')
        : ''
      return client.get(`/crag/cache${params ? '?' + params : ''}`)
    },

    deleteCache: (snapshot_ids) => {
      const params = snapshot_ids.map((id) => `snapshot_ids=${id}`).join('&')
      return client.delete(`/crag/cache/snapshots?${params}`)
    },
  },

  // ── Ollama ─────────────────────────────────────────────────────────────────
  ollama: {
    health: () => client.get('/ollama/health'),
    llmModels: () => client.get('/ollama/models/llm'),
    embeddingModels: () => client.get('/ollama/models/embedding'),
    pullLlm: (model) => client.post('/ollama/models/llm/pull', { model }),
    pullEmbedding: (model) => client.post('/ollama/models/embedding/pull', { model }),
  },
}

export default api
