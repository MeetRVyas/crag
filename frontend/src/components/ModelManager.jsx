import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Download, CheckCircle2, AlertCircle, RefreshCw, Cpu, Layers } from 'lucide-react'
import api from '../api/client'

const BACKDROP = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  exit:    { opacity: 0 },
}
const MODAL = {
  initial: { opacity: 0, scale: 0.96, y: 12 },
  animate: { opacity: 1, scale: 1,    y: 0  },
  exit:    { opacity: 0, scale: 0.96, y: 12 },
  transition: { duration: 0.22, ease: [0.22, 1, 0.36, 1] },
}

/* ── Model row ────────────────────────────────────────── */
function ModelRow({ model, type, onPull }) {
  const [pulling,  setPulling]  = useState(false)
  const [pullDone, setPullDone] = useState(false)
  const [error,    setError]    = useState(null)

  const handlePull = async () => {
    if (pulling || model.installed) return
    setPulling(true)
    setError(null)
    try {
      await (type === 'llm'
        ? api.ollama.pullLlm(model.name)
        : api.ollama.pullEmbedding(model.name))
      setPullDone(true)
      onPull()
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Pull failed')
    } finally {
      setPulling(false)
    }
  }

  const installed = model.installed || pullDone

  return (
    <motion.div
      layout
      className="flex items-center gap-3 px-4 py-3 rounded-md"
      style={{ background: 'var(--c-surface-hi)', border: '1px solid var(--c-border)' }}
    >
      {/* Status dot */}
      <div
        className="w-2 h-2 rounded-full flex-shrink-0"
        style={{
          background: installed ? 'var(--c-green)' : 'var(--c-muted)',
          boxShadow: installed ? '0 0 6px var(--c-green)' : 'none',
        }}
      />

      {/* Info */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-mono text-[12px] text-text truncate">{model.name}</span>
          {model.is_default && (
            <span
              className="text-[9px] font-mono px-1.5 py-0.5 rounded-sm flex-shrink-0"
              style={{ background: 'var(--c-primary-10)', color: 'var(--c-primary)' }}
            >
              DEFAULT
            </span>
          )}
        </div>
        <p className="font-mono text-[10px] text-muted">
          {installed ? 'Installed' : 'Not installed'}{error ? ` · ${error}` : ''}
        </p>
      </div>

      {/* Pull button */}
      {!installed && (
        <button
          onClick={handlePull}
          disabled={pulling}
          className="flex items-center gap-1.5 btn-ghost text-[11px] py-1 px-2.5"
          title="Pull model"
        >
          {pulling
            ? <RefreshCw size={11} className="animate-spin" />
            : <Download size={11} />}
          {pulling ? 'Pulling…' : 'Pull'}
        </button>
      )}
      {installed && <CheckCircle2 size={14} style={{ color: 'var(--c-green)', flexShrink: 0 }} />}
    </motion.div>
  )
}

/* ── Health status ────────────────────────────────────── */
function HealthBadge({ health }) {
  if (!health) return null
  const ok = health.status === 'ok'
  return (
    <div
      className="flex items-center gap-1.5 font-mono text-[10px] px-2.5 py-1 rounded-sm"
      style={{
        background: ok ? 'var(--c-green-10)' : 'var(--c-red-10)',
        color:      ok ? 'var(--c-green)'    : 'var(--c-red)',
        border:     `1px solid ${ok ? 'rgba(0,200,150,0.2)' : 'rgba(240,62,27,0.2)'}`,
      }}
    >
      <div
        className="w-1.5 h-1.5 rounded-full"
        style={{ background: ok ? 'var(--c-green)' : 'var(--c-red)' }}
      />
      Ollama {ok ? 'online' : 'offline'}
      {ok && <span className="text-muted">· {health.installed_models?.length ?? 0} models</span>}
    </div>
  )
}

/* ── Model Manager Modal ──────────────────────────────── */
export default function ModelManager({ onClose }) {
  const [tab,       setTab]       = useState('llm')
  const [llmModels, setLlmModels] = useState([])
  const [embModels, setEmbModels] = useState([])
  const [health,    setHealth]    = useState(null)
  const [loading,   setLoading]   = useState(true)
  const [error,     setError]     = useState(null)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [h, l, e] = await Promise.all([
        api.ollama.health(),
        api.ollama.llmModels(),
        api.ollama.embeddingModels(),
      ])
      setHealth(h.data)
      setLlmModels(l.data.models ?? [])
      setEmbModels(e.data.models ?? [])
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Failed to reach Ollama')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { load() }, [load])

  const models = tab === 'llm' ? llmModels : embModels

  return (
    <motion.div
      {...BACKDROP}
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(8,9,14,0.85)', backdropFilter: 'blur(8px)' }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <motion.div
        {...MODAL}
        className="w-full max-w-lg rounded-xl overflow-hidden"
        style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)' }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-5 py-4 border-b"
          style={{ borderColor: 'var(--c-border)' }}
        >
          <div>
            <h2
              className="text-xl tracking-widest"
              style={{ fontFamily: 'var(--f-display)', letterSpacing: '0.12em' }}
            >
              MODEL MANAGER
            </h2>
            <p className="font-mono text-[10px] text-muted mt-0.5">
              Manage Ollama LLM and embedding models
            </p>
          </div>
          <div className="flex items-center gap-3">
            <HealthBadge health={health} />
            <button onClick={onClose} className="text-muted hover:text-text transition-colors">
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div
          className="flex border-b"
          style={{ borderColor: 'var(--c-border)' }}
        >
          {[
            { id: 'llm', label: 'LLM Models',       icon: <Cpu size={12} /> },
            { id: 'emb', label: 'Embedding Models',  icon: <Layers size={12} /> },
          ].map(({ id, label, icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className="flex items-center gap-2 px-5 py-3 font-condensed font-semibold text-[12px] uppercase tracking-widest transition-all relative"
              style={{
                color: tab === id ? 'var(--c-primary)' : 'var(--c-muted)',
                letterSpacing: '0.1em',
              }}
            >
              {icon} {label}
              {tab === id && (
                <motion.div
                  layoutId="tab-indicator"
                  className="absolute bottom-0 left-0 right-0 h-[2px]"
                  style={{ background: 'var(--c-primary)' }}
                />
              )}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-5 space-y-3 max-h-96 overflow-y-auto">
          {loading && (
            <div className="flex items-center justify-center py-12">
              <RefreshCw size={18} className="animate-spin text-primary" />
            </div>
          )}
          {error && !loading && (
            <div
              className="flex items-center gap-2 px-4 py-3 rounded-md font-mono text-[11px] text-red"
              style={{ background: 'var(--c-red-10)', border: '1px solid rgba(240,62,27,0.2)' }}
            >
              <AlertCircle size={12} />
              {error}
            </div>
          )}
          {!loading && !error && models.map((m) => (
            <ModelRow
              key={m.name}
              model={m}
              type={tab}
              onPull={load}
            />
          ))}
        </div>

        {/* Footer */}
        <div
          className="px-5 py-3 border-t flex justify-between items-center"
          style={{ borderColor: 'var(--c-border)', background: 'var(--c-bg)' }}
        >
          <p className="font-mono text-[10px] text-muted">
            Models not listed here can be entered manually in the chat bar
          </p>
          <button onClick={load} className="btn-ghost flex items-center gap-1.5">
            <RefreshCw size={11} />
            Refresh
          </button>
        </div>
      </motion.div>
    </motion.div>
  )
}
