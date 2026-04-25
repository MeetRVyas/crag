import { useState } from 'react'
import { motion } from 'framer-motion'
import { X, Eye, EyeOff, Key, CheckCircle2, AlertCircle, Lock } from 'lucide-react'
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

const PROVIDERS = [
  {
    id:          'groq',
    label:       'Groq',
    placeholder: 'gsk_…',
    hint:        'Required for Groq LLM provider',
  },
  {
    id:          'anthropic',
    label:       'Anthropic',
    placeholder: 'sk-ant-…',
    hint:        'Required for Claude models',
  },
  {
    id:          'google',
    label:       'Google',
    placeholder: 'AIza…',
    hint:        'Required for Gemini LLM + embeddings',
  },
  {
    id:          'huggingface',
    label:       'HuggingFace',
    placeholder: 'hf_…',
    hint:        'Required for HF API provider',
  },
  {
    id:          'tavily',
    label:       'Tavily',
    placeholder: 'tvly-…',
    hint:        'Enables web search fallback',
  },
]

/* ── Key input row ────────────────────────────────────── */
function KeyRow({ provider, value, onChange }) {
  const [visible, setVisible] = useState(false)

  return (
    <div>
      <div className="flex items-center justify-between mb-1.5">
        <label className="font-mono text-[11px] text-dim">{provider.label}</label>
        <span className="font-mono text-[9px] text-muted">{provider.hint}</span>
      </div>
      <div className="relative">
        <input
          type={visible ? 'text' : 'password'}
          className="input-base pr-9"
          placeholder={provider.placeholder}
          value={value}
          onChange={(e) => onChange(provider.id, e.target.value)}
          autoComplete="off"
          spellCheck={false}
        />
        <button
          type="button"
          onClick={() => setVisible((v) => !v)}
          className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted hover:text-dim transition-colors"
        >
          {visible ? <EyeOff size={12} /> : <Eye size={12} />}
        </button>
      </div>
    </div>
  )
}

/* ── API Key Modal ────────────────────────────────────── */
export default function ApiKeyModal({ onClose }) {
  const [keys,    setKeys]    = useState({})
  const [saving,  setSaving]  = useState(false)
  const [success, setSuccess] = useState(false)
  const [error,   setError]   = useState(null)

  const handleChange = (id, value) =>
    setKeys((prev) => ({ ...prev, [id]: value }))

  const handleSave = async () => {
    // Only include non-empty keys
    const toSave = Object.fromEntries(
      Object.entries(keys).filter(([, v]) => v.trim().length >= 10)
    )
    if (!Object.keys(toSave).length) {
      setError('Enter at least one API key (min 10 chars)')
      return
    }

    setSaving(true)
    setError(null)

    try {
      await api.auth.setKeys(toSave)
      setSuccess(true)
      setTimeout(() => { setSuccess(false); onClose() }, 1200)
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Failed to save keys')
    } finally {
      setSaving(false)
    }
  }

  return (
    <motion.div
      {...BACKDROP}
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(8,9,14,0.85)', backdropFilter: 'blur(8px)' }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <motion.div
        {...MODAL}
        className="w-full max-w-md rounded-xl overflow-hidden"
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
              API KEYS
            </h2>
            <p className="font-mono text-[10px] text-muted mt-0.5 flex items-center gap-1">
              <Lock size={9} />
              Encrypted and stored in your session · never persisted to disk
            </p>
          </div>
          <button onClick={onClose} className="text-muted hover:text-text transition-colors">
            <X size={16} />
          </button>
        </div>

        {/* Keys */}
        <div className="p-5 space-y-4">
          {PROVIDERS.map((p) => (
            <KeyRow
              key={p.id}
              provider={p}
              value={keys[p.id] ?? ''}
              onChange={handleChange}
            />
          ))}

          {/* Security note */}
          <div
            className="flex items-start gap-2 px-3 py-2.5 rounded-md font-mono text-[10px] text-muted"
            style={{ background: 'var(--c-bg)', border: '1px solid var(--c-border)' }}
          >
            <Key size={10} className="flex-shrink-0 mt-0.5" />
            Keys are encrypted with Fernet AES-128 and stored only in Redis for
            your session duration. They are never written to disk or logged.
          </div>

          {/* Error */}
          {error && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex items-center gap-2 px-3 py-2 rounded-md font-mono text-[11px] text-red"
              style={{ background: 'var(--c-red-10)', border: '1px solid rgba(240,62,27,0.2)' }}
            >
              <AlertCircle size={11} />
              {error}
            </motion.div>
          )}
        </div>

        {/* Footer */}
        <div
          className="px-5 py-4 border-t flex justify-end gap-3"
          style={{ borderColor: 'var(--c-border)', background: 'var(--c-bg)' }}
        >
          <button onClick={onClose} className="btn-ghost">Cancel</button>
          <button
            onClick={handleSave}
            disabled={saving || success}
            className="btn-primary flex items-center gap-2"
          >
            {success
              ? <><CheckCircle2 size={13} /> Saved!</>
              : saving
              ? 'Saving…'
              : <><Key size={13} /> Save Keys</>}
          </button>
        </div>
      </motion.div>
    </motion.div>
  )
}
