import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Send, Globe, Database, CheckCircle2,
  AlertTriangle, XCircle, Loader2, ChevronDown,
  Sparkles, MessageSquare, Copy, Check,
} from 'lucide-react'
import api from '../api/client'

/* ── Constants ────────────────────────────────────────── */
const LLM_PROVIDERS = [
  { id: 'ollama',            label: 'Ollama',          local: true  },
  { id: 'groq',              label: 'Groq',             local: false },
  { id: 'anthropic',         label: 'Anthropic',        local: false },
  { id: 'google',            label: 'Google Gemini',    local: false },
  { id: 'huggingface',       label: 'HuggingFace API',  local: false },
  { id: 'huggingface_local', label: 'HF Local',         local: true  },
]

const EMB_PROVIDERS = [
  { id: 'ollama',      label: 'Ollama',      local: true  },
  { id: 'google',      label: 'Google',      local: false },
  { id: 'huggingface', label: 'HuggingFace', local: true  },
]

/* ── Copy button ──────────────────────────────────────── */
function CopyButton({ text, className = '' }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async (e) => {
    e.stopPropagation()
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch (_) {}
  }

  return (
    <button
      onClick={handleCopy}
      title={copied ? 'Copied!' : 'Copy'}
      className={`flex items-center gap-1 transition-all duration-150 ${className}`}
      style={{
        color: copied ? 'var(--c-green)' : 'var(--c-muted)',
        fontFamily: 'var(--f-mono)',
        fontSize: 10,
      }}
    >
      {copied
        ? <><Check size={11} /><span>Copied</span></>
        : <><Copy size={11} /><span>Copy</span></>
      }
    </button>
  )
}

/* ── Markdown renderer ────────────────────────────────── */
const markdownComponents = {
  // Paragraphs — no extra margin inside bubbles
  p: ({ children }) => (
    <p style={{ margin: '0 0 0.6em', lineHeight: 1.7 }} className="last:mb-0">
      {children}
    </p>
  ),

  // Headings
  h1: ({ children }) => (
    <h1 style={{
      fontFamily: 'var(--f-display)', fontSize: 22, letterSpacing: '0.06em',
      color: 'var(--c-text)', margin: '0.8em 0 0.4em', lineHeight: 1.2,
    }}>{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 style={{
      fontFamily: 'var(--f-display)', fontSize: 17, letterSpacing: '0.06em',
      color: 'var(--c-text)', margin: '0.8em 0 0.3em', lineHeight: 1.2,
    }}>{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 style={{
      fontFamily: 'var(--f-condensed)', fontWeight: 600, fontSize: 14,
      letterSpacing: '0.05em', color: 'var(--c-dim)', margin: '0.6em 0 0.25em',
      textTransform: 'uppercase',
    }}>{children}</h3>
  ),

  // Inline code
  code: ({ inline, children, ...props }) => {
    if (inline) {
      return (
        <code
          style={{
            fontFamily: 'var(--f-mono)', fontSize: '0.85em',
            background: 'var(--c-surface-hi)', color: 'var(--c-primary)',
            padding: '1px 6px', borderRadius: 4,
            border: '1px solid var(--c-border)',
          }}
          {...props}
        >
          {children}
        </code>
      )
    }
    return <code style={{ fontFamily: 'var(--f-mono)', fontSize: 12 }} {...props}>{children}</code>
  },

  // Code blocks — with its own copy button
  pre: ({ children }) => {
    const codeEl = children?.props
    const raw = codeEl?.children ?? ''
    const lang = (codeEl?.className ?? '').replace('language-', '') || null

    return (
      <div style={{
        position: 'relative', margin: '0.6em 0',
        background: 'var(--c-bg)', border: '1px solid var(--c-border)',
        borderRadius: 6, overflow: 'hidden',
      }}>
        {/* Top bar */}
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '4px 10px',
          borderBottom: '1px solid var(--c-border)',
          background: 'var(--c-surface)',
        }}>
          <span style={{ fontFamily: 'var(--f-mono)', fontSize: 10, color: 'var(--c-muted)' }}>
            {lang ?? 'code'}
          </span>
          <CopyButton text={String(raw).replace(/\n$/, '')} />
        </div>
        <pre style={{
          margin: 0, padding: '10px 12px', overflowX: 'auto',
          fontFamily: 'var(--f-mono)', fontSize: 12, lineHeight: 1.6,
          color: 'var(--c-dim)',
        }}>
          {children}
        </pre>
      </div>
    )
  },

  // Lists
  ul: ({ children }) => (
    <ul style={{ margin: '0.4em 0', paddingLeft: 20, lineHeight: 1.7 }}>{children}</ul>
  ),
  ol: ({ children }) => (
    <ol style={{ margin: '0.4em 0', paddingLeft: 20, lineHeight: 1.7 }}>{children}</ol>
  ),
  li: ({ children }) => (
    <li style={{ margin: '0.15em 0', color: 'var(--c-text)' }}>{children}</li>
  ),

  // Blockquote
  blockquote: ({ children }) => (
    <blockquote style={{
      margin: '0.6em 0', padding: '6px 12px',
      borderLeft: '3px solid var(--c-primary)',
      background: 'var(--c-surface-hi)',
      borderRadius: '0 4px 4px 0',
      color: 'var(--c-dim)',
      fontStyle: 'italic',
    }}>
      {children}
    </blockquote>
  ),

  // Horizontal rule
  hr: () => (
    <hr style={{ border: 'none', borderTop: '1px solid var(--c-border)', margin: '0.8em 0' }} />
  ),

  // Links
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      style={{ color: 'var(--c-primary)', textDecoration: 'underline', textUnderlineOffset: 3 }}
    >
      {children}
    </a>
  ),

  // Strong / em
  strong: ({ children }) => (
    <strong style={{ fontWeight: 600, color: 'var(--c-text)' }}>{children}</strong>
  ),
  em: ({ children }) => (
    <em style={{ fontStyle: 'italic', color: 'var(--c-dim)' }}>{children}</em>
  ),

  // Tables
  table: ({ children }) => (
    <div style={{ overflowX: 'auto', margin: '0.6em 0' }}>
      <table style={{
        borderCollapse: 'collapse', width: '100%',
        fontFamily: 'var(--f-mono)', fontSize: 12,
      }}>
        {children}
      </table>
    </div>
  ),
  thead: ({ children }) => (
    <thead style={{ background: 'var(--c-surface-hi)' }}>{children}</thead>
  ),
  th: ({ children }) => (
    <th style={{
      padding: '6px 12px', textAlign: 'left',
      border: '1px solid var(--c-border)',
      color: 'var(--c-dim)', fontWeight: 600,
      letterSpacing: '0.05em', textTransform: 'uppercase', fontSize: 10,
    }}>
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td style={{
      padding: '5px 12px',
      border: '1px solid var(--c-border)',
      color: 'var(--c-text)',
    }}>
      {children}
    </td>
  ),
}

/* ── Verdict badge ────────────────────────────────────── */
function VerdictBadge({ verdict }) {
  const map = {
    CORRECT:   { cls: 'badge-correct',   icon: <CheckCircle2  size={10} />, label: 'CORRECT'   },
    AMBIGUOUS: { cls: 'badge-ambiguous', icon: <AlertTriangle size={10} />, label: 'AMBIGUOUS' },
    INCORRECT: { cls: 'badge-incorrect', icon: <XCircle       size={10} />, label: 'INCORRECT' },
  }
  const { cls, icon, label } = map[verdict] ?? map.AMBIGUOUS
  return (
    <span className={`inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-sm ${cls}`}>
      {icon} {label}
    </span>
  )
}

/* ── Select dropdown ──────────────────────────────────── */
function Select({ value, onChange, options, className = '' }) {
  return (
    <div className={`relative ${className}`}>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="input-base pr-7 appearance-none cursor-pointer text-[11px] py-1.5"
        style={{ paddingRight: 28 }}
      >
        {options.map((o) => (
          <option key={o.id ?? o} value={o.id ?? o}>
            {o.label ?? o}
          </option>
        ))}
      </select>
      <ChevronDown
        size={11}
        className="absolute right-2 top-1/2 -translate-y-1/2 text-muted pointer-events-none"
      />
    </div>
  )
}

/* ── Model config bar ─────────────────────────────────── */
function ModelBar({ config, setConfig, ollamaLLMModels, ollamaEmbModels }) {
  const llmOptions = config.llmProvider === 'ollama'
    ? ollamaLLMModels.map((m) => ({ id: m, label: m }))
    : [{ id: config.llmModel || 'enter-model-name', label: config.llmModel || 'Enter model name' }]

  const embOptions = config.embProvider === 'ollama'
    ? ollamaEmbModels.map((m) => ({ id: m, label: m }))
    : [{ id: config.embModel || 'enter-model', label: config.embModel || 'Enter model' }]

  return (
    <div
      className="flex-shrink-0 flex flex-wrap items-center gap-2 px-4 py-2 border-t"
      style={{ borderColor: 'var(--c-border)', background: 'var(--c-surface)' }}
    >
      <div className="flex items-center gap-1.5">
        <span className="panel-label">LLM</span>
        <Select
          value={config.llmProvider}
          onChange={(v) => setConfig((c) => ({ ...c, llmProvider: v, llmModel: '' }))}
          options={LLM_PROVIDERS}
        />
        {config.llmProvider === 'ollama' ? (
          <Select
            value={config.llmModel}
            onChange={(v) => setConfig((c) => ({ ...c, llmModel: v }))}
            options={llmOptions.length ? llmOptions : [{ id: '', label: 'No models' }]}
          />
        ) : (
          <input
            className="input-base text-[11px] py-1.5 w-40"
            placeholder="Model name…"
            value={config.llmModel}
            onChange={(e) => setConfig((c) => ({ ...c, llmModel: e.target.value }))}
          />
        )}
      </div>

      <div className="w-px h-4 bg-border" />

      <div className="flex items-center gap-1.5">
        <span className="panel-label">EMB</span>
        <Select
          value={config.embProvider}
          onChange={(v) => setConfig((c) => ({ ...c, embProvider: v, embModel: '' }))}
          options={EMB_PROVIDERS}
        />
        {config.embProvider === 'ollama' ? (
          <Select
            value={config.embModel}
            onChange={(v) => setConfig((c) => ({ ...c, embModel: v }))}
            options={embOptions.length ? embOptions : [{ id: '', label: 'No models' }]}
          />
        ) : (
          <input
            className="input-base text-[11px] py-1.5 w-40"
            placeholder="Model name…"
            value={config.embModel}
            onChange={(e) => setConfig((c) => ({ ...c, embModel: e.target.value }))}
          />
        )}
      </div>
    </div>
  )
}

/* ── Message bubble ───────────────────────────────────── */
function MessageBubble({ msg }) {
  const [hovered, setHovered] = useState(false)
  const isUser = msg.role === 'user'

  if (isUser) {
    return (
      <div
        className="flex justify-end msg-in"
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        <div className="max-w-[70%] flex flex-col gap-1 items-end">
          {/* Copy button — appears above bubble on hover */}
          <AnimatePresence>
            {hovered && (
              <motion.div
                initial={{ opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 4 }}
                transition={{ duration: 0.12 }}
              >
                <CopyButton text={msg.content} />
              </motion.div>
            )}
          </AnimatePresence>

          <div
            className="px-4 py-3 rounded-lg rounded-br-sm"
            style={{
              background: 'linear-gradient(135deg, var(--c-primary) 0%, #005F8A 100%)',
              color: 'white',
              fontFamily: 'var(--f-body)',
              fontSize: 14,
              lineHeight: 1.6,
            }}
          >
            {/* User content — also render markdown so pasted markdown looks right */}
            <div className="prose-user">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  ...markdownComponents,
                  // Override link/strong/em/code colours for white-on-blue bubble
                  a: ({ href, children }) => (
                    <a href={href} target="_blank" rel="noopener noreferrer"
                      style={{ color: 'rgba(255,255,255,0.85)', textDecoration: 'underline', textUnderlineOffset: 3 }}>
                      {children}
                    </a>
                  ),
                  strong: ({ children }) => (
                    <strong style={{ fontWeight: 700, color: 'white' }}>{children}</strong>
                  ),
                  code: ({ inline, children, ...props }) =>
                    inline
                      ? <code style={{
                          fontFamily: 'var(--f-mono)', fontSize: '0.85em',
                          background: 'rgba(0,0,0,0.25)', color: 'rgba(255,255,255,0.9)',
                          padding: '1px 6px', borderRadius: 4,
                        }} {...props}>{children}</code>
                      : <code style={{ fontFamily: 'var(--f-mono)', fontSize: 12 }} {...props}>{children}</code>,
                  p: ({ children }) => (
                    <p style={{ margin: '0 0 0.5em', lineHeight: 1.7 }} className="last:mb-0">{children}</p>
                  ),
                }}
              >
                {msg.content}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Assistant message
  return (
    <div
      className="flex flex-col gap-2 msg-in"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      <div
        className="max-w-[85%] px-4 py-3 rounded-lg rounded-tl-sm relative"
        style={{
          background: 'var(--c-surface)',
          border: '1px solid var(--c-border)',
          fontFamily: 'var(--f-body)',
          fontSize: 14,
          color: 'var(--c-text)',
        }}
      >
        {msg.loading ? (
          <div className="flex items-center gap-2 text-dim">
            <Loader2 size={13} className="animate-spin text-primary" />
            <span className="font-mono text-[11px]">Running pipeline…</span>
          </div>
        ) : (
          <>
            {/* Copy button — top-right corner of bubble */}
            <AnimatePresence>
              {hovered && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.12 }}
                  style={{ position: 'absolute', top: 8, right: 10 }}
                >
                  <CopyButton text={msg.content} />
                </motion.div>
              )}
            </AnimatePresence>

            <div className="markdown-body pr-12">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
                {msg.content}
              </ReactMarkdown>
            </div>
          </>
        )}
      </div>

      {/* Metadata row */}
      {!msg.loading && (
        <div className="flex flex-wrap items-center gap-2 pl-1">
          {msg.verdict && <VerdictBadge verdict={msg.verdict} />}
          {msg.web_search_used && (
            <span
              className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-sm"
              style={{ background: 'var(--c-red-10)', color: 'var(--c-red)', border: '1px solid rgba(240,62,27,0.2)' }}
            >
              <Globe size={10} /> Web search
            </span>
          )}
          {msg.cached && (
            <span
              className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-sm"
              style={{ background: 'var(--c-yellow-10)', color: 'var(--c-yellow)', border: '1px solid rgba(255,208,0,0.2)' }}
            >
              <Database size={10} /> Cached
            </span>
          )}
          {msg.snapshot_id && (
            <span className="text-[9px] font-mono text-muted">
              snap · {msg.snapshot_id.slice(-8)}
            </span>
          )}
        </div>
      )}
    </div>
  )
}

/* ── Empty state ──────────────────────────────────────── */
function EmptyState() {
  return (
    <motion.div
      className="flex-1 flex flex-col items-center justify-center text-center px-8"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 0.3 }}
    >
      <div
        className="w-16 h-16 rounded-2xl flex items-center justify-center mb-5"
        style={{
          background: 'var(--c-surface)',
          border: '1px solid var(--c-border)',
          boxShadow: '0 0 32px var(--c-primary-10)',
        }}
      >
        <Sparkles size={24} className="text-primary" />
      </div>
      <h2
        className="text-3xl mb-2"
        style={{ fontFamily: 'var(--f-display)', letterSpacing: '0.08em' }}
      >
        Ask anything
      </h2>
      <p className="text-dim text-sm leading-relaxed max-w-sm">
        Upload and index your PDFs, then ask questions. The pipeline retrieves,
        evaluates, and generates answers — with web search when your documents
        aren't enough.
      </p>
      <div className="flex gap-3 mt-6 flex-wrap justify-center">
        {[
          'What does this document say about…',
          'Summarise the key findings',
          'Compare X and Y from the text',
        ].map((p) => (
          <span
            key={p}
            className="text-[11px] font-mono text-muted px-3 py-1.5 rounded-sm"
            style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)' }}
          >
            "{p}"
          </span>
        ))}
      </div>
    </motion.div>
  )
}

/* ── Chat Panel ───────────────────────────────────────── */
export default function ChatPanel({
  retrieverConfig,
  setRetrieverConfig,
  onPipelineStart,
  onPipelineEvent,
  pipelineActive,
}) {
  const [messages,  setMessages]  = useState([])
  const [input,     setInput]     = useState('')
  const [sending,   setSending]   = useState(false)
  const [ollamaLLM, setOllamaLLM] = useState([])
  const [ollamaEmb, setOllamaEmb] = useState([])
  const bottomRef   = useRef(null)
  const inputRef    = useRef(null)
  const eventSrcRef = useRef(null)

  useEffect(() => {
    api.ollama.llmModels().then((r) => {
      const installed = (r.data.models ?? []).filter((m) => m.installed).map((m) => m.name)
      setOllamaLLM(installed)
      if (installed.length && !retrieverConfig.llmModel)
        setRetrieverConfig((c) => ({ ...c, llmModel: installed[0] }))
    }).catch(() => {})

    api.ollama.embeddingModels().then((r) => {
      const installed = (r.data.models ?? []).filter((m) => m.installed).map((m) => m.name)
      setOllamaEmb(installed)
      if (installed.length && !retrieverConfig.embModel)
        setRetrieverConfig((c) => ({ ...c, embModel: installed[0] }))
    }).catch(() => {})
  }, []) // eslint-disable-line

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const openSSE = useCallback(() => {
    eventSrcRef.current?.close()
    const token = localStorage.getItem('crag_token')
    const base  = import.meta.env.VITE_API_URL ?? '/api'
    const src   = new EventSource(`${base}/crag/status?token=${token}`)
    eventSrcRef.current = src

    src.onmessage = (e) => {
      if (e.data?.startsWith(':')) return
      try {
        const ev = JSON.parse(e.data)
        onPipelineEvent(ev)
        if (ev.complete) { src.close(); eventSrcRef.current = null }
      } catch (_) {}
    }
    src.onerror = () => {
      src.close()
      eventSrcRef.current = null
      onPipelineEvent({ complete: true, verdict: 'ERROR' })
    }
  }, [onPipelineEvent])

  const handleSend = async () => {
    const q = input.trim()
    if (!q || sending) return

    setSending(true)
    setInput('')
    onPipelineStart()

    const userMsg = { id: Date.now(),     role: 'user',      content: q }
    const loadMsg = { id: Date.now() + 1, role: 'assistant', loading: true }
    setMessages((prev) => [...prev, userMsg, loadMsg])

    openSSE()

    try {
      const res = await api.crag.chat({
        question:           q,
        llm_provider:       retrieverConfig.llmProvider,
        llm_model:          retrieverConfig.llmModel,
        embedding_provider: retrieverConfig.embProvider,
        embedding_model:    retrieverConfig.embModel,
      })

      const { answer, verdict, web_search_used, cached, snapshot_id } = res.data
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadMsg.id
            ? { ...m, loading: false, content: answer, verdict, web_search_used, cached, snapshot_id }
            : m
        )
      )
    } catch (err) {
      const detail = err.response?.data?.detail ?? 'An error occurred. Check your documents are indexed.'
      setMessages((prev) =>
        prev.map((m) =>
          m.id === loadMsg.id
            ? { ...m, loading: false, content: `⚠ ${detail}`, verdict: 'INCORRECT' }
            : m
        )
      )
      onPipelineEvent({ complete: true, verdict: 'ERROR' })
    } finally {
      setSending(false)
      inputRef.current?.focus()
    }
  }

  const handleKey = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="p-4 space-y-5 max-w-3xl mx-auto">
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25 }}
                >
                  <MessageBubble msg={msg} />
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input area */}
      <div
        className="flex-shrink-0 border-t"
        style={{ borderColor: 'var(--c-border)', background: 'var(--c-bg)' }}
      >
        <div className="max-w-3xl mx-auto p-4">
          <div
            className="flex items-end gap-3 rounded-lg p-3"
            style={{
              background: 'var(--c-surface)',
              border: sending ? '1px solid var(--c-primary)' : '1px solid var(--c-border)',
              boxShadow: sending ? '0 0 0 1px var(--c-primary), 0 0 16px var(--c-primary-10)' : 'none',
              transition: 'border-color 0.15s, box-shadow 0.15s',
            }}
          >
            <MessageSquare size={15} className="text-muted mb-1.5 flex-shrink-0" />
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKey}
              placeholder="Ask about your documents…"
              disabled={sending}
              rows={1}
              className="flex-1 bg-transparent resize-none outline-none text-sm text-text placeholder-muted font-body"
              style={{ fontFamily: 'var(--f-body)', maxHeight: 120, lineHeight: 1.6 }}
              onInput={(e) => {
                e.target.style.height = 'auto'
                e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px'
              }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || sending}
              className="flex-shrink-0 w-8 h-8 rounded-md flex items-center justify-center transition-all duration-150"
              style={{
                background: input.trim() && !sending ? 'var(--c-primary)' : 'var(--c-surface-hi)',
                color: input.trim() && !sending ? 'white' : 'var(--c-muted)',
                boxShadow: input.trim() && !sending ? '0 0 12px var(--c-primary-20)' : 'none',
              }}
            >
              {sending
                ? <Loader2 size={14} className="animate-spin" />
                : <Send size={14} />
              }
            </button>
          </div>
          <p className="mt-1.5 text-center font-mono text-[10px] text-muted">
            Enter to send · Shift+Enter for newline · Pipeline runs in 6 stages
          </p>
        </div>
      </div>

      {/* Model config bar */}
      <ModelBar
        config={retrieverConfig}
        setConfig={setRetrieverConfig}
        ollamaLLMModels={ollamaLLM}
        ollamaEmbModels={ollamaEmb}
      />
    </div>
  )
}