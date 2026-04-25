import { useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { X, CheckCircle2, Loader2, Circle } from 'lucide-react'

/* ── Pipeline stage definitions ───────────────────────── */
const STAGES = [
  { key: 'retrieve', label: 'RETRIEVE', desc: 'HyDE + FAISS + BM25',       color: 'var(--c-primary)' },
  { key: 'evaluate', label: 'EVALUATE', desc: 'Relevance scoring 0–1',       color: 'var(--c-primary)' },
  { key: 'rewrite',  label: 'REWRITE',  desc: 'Query rewrite for web',       color: 'var(--c-yellow)'  },
  { key: 'research', label: 'RESEARCH', desc: 'Tavily web search',            color: 'var(--c-yellow)'  },
  { key: 'refine',   label: 'REFINE',   desc: 'Sentence-level filtering',    color: 'var(--c-primary)' },
  { key: 'generate', label: 'GENERATE', desc: 'LLM answer synthesis',        color: 'var(--c-green)'   },
]

function getStageStatus(stageKey, events, activeStep) {
  const idx     = STAGES.findIndex((s) => s.key === stageKey)
  const active  = STAGES.findIndex((s) => s.key === activeStep)

  if (active < 0) {
    // pipeline completed — check if this stage appeared in events
    const appeared = events.some((e) => e.step === stageKey)
    return appeared ? 'done' : 'skipped'
  }
  if (idx < active)  return 'done'
  if (idx === active) return 'active'
  return 'pending'
}

/* ── Stage node ───────────────────────────────────────── */
function StageNode({ stage, status, event, isLast }) {
  const { label, desc, color } = stage

  const dot = {
    done:    <CheckCircle2 size={14} style={{ color: 'var(--c-green)' }} />,
    active:  <Loader2 size={14} className="animate-spin" style={{ color }} />,
    pending: <Circle size={14} style={{ color: 'var(--c-muted)' }} />,
    skipped: <Circle size={14} style={{ color: 'var(--c-border-hi)' }} />,
  }[status]

  return (
    <div className="flex gap-3">
      {/* Timeline */}
      <div className="flex flex-col items-center flex-shrink-0">
        <motion.div
          initial={{ scale: 0.6, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="w-6 h-6 flex items-center justify-center"
        >
          {dot}
        </motion.div>
        {!isLast && (
          <div
            className="w-px flex-1 mt-1"
            style={{
              background: status === 'done'
                ? 'linear-gradient(to bottom, var(--c-green), var(--c-border))'
                : 'var(--c-border)',
              minHeight: 16,
            }}
          />
        )}
      </div>

      {/* Content */}
      <div className="pb-4 min-w-0">
        <div className="flex items-baseline gap-2">
          <span
            className="font-display text-sm tracking-widest"
            style={{
              color: status === 'done'
                ? 'var(--c-text)'
                : status === 'active'
                ? color
                : 'var(--c-muted)',
              fontFamily: 'var(--f-display)',
              fontSize: 13,
            }}
          >
            {label}
          </span>
          <span className="font-mono text-[9px] text-muted">{desc}</span>
        </div>
        <AnimatePresence>
          {event && (
            <motion.p
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="font-mono text-[10px] mt-1 leading-relaxed"
              style={{ color: status === 'active' ? 'var(--c-dim)' : 'var(--c-muted)' }}
            >
              {event.message}
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

/* ── Verdict display ──────────────────────────────────── */
function VerdictDisplay({ verdict }) {
  const map = {
    CORRECT:   { color: 'var(--c-green)',  bg: 'var(--c-green-10)', label: 'CORRECT — Documents answered' },
    AMBIGUOUS: { color: 'var(--c-yellow)', bg: 'var(--c-yellow-10)', label: 'AMBIGUOUS — Docs + Web combined' },
    INCORRECT: { color: 'var(--c-red)',    bg: 'var(--c-red-10)',   label: 'INCORRECT — Web search used' },
    ERROR:     { color: 'var(--c-red)',    bg: 'var(--c-red-10)',   label: 'ERROR — Pipeline failed' },
  }
  const { color, bg, label } = map[verdict] ?? map.AMBIGUOUS

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className="px-3 py-2.5 rounded-md font-mono text-[11px] text-center"
      style={{ background: bg, color, border: `1px solid ${color}30` }}
    >
      {label}
    </motion.div>
  )
}

/* ── Log stream ───────────────────────────────────────── */
function LogLine({ event }) {
  const ts = new Date(event.timestamp)
  const time = ts.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })

  return (
    <motion.div
      initial={{ opacity: 0, x: -4 }}
      animate={{ opacity: 1, x: 0 }}
      className="flex gap-2 font-mono text-[10px] py-0.5"
    >
      <span className="text-muted flex-shrink-0">{time}</span>
      <span
        className="flex-shrink-0 uppercase"
        style={{ color: 'var(--c-primary)', width: 60 }}
      >
        {event.step}
      </span>
      <span className="text-dim truncate">{event.message}</span>
    </motion.div>
  )
}

/* ── Pipeline Drawer ──────────────────────────────────── */
export default function PipelineDrawer({ events, active, onClose }) {
  const logRef = useRef(null)

  // Determine the currently active step from events
  const stepEvents = events.filter((e) => e.step && !e.complete)
  const activeStep = active && stepEvents.length
    ? stepEvents[stepEvents.length - 1].step
    : null

  // Find the terminal event
  const completeEvent = events.find((e) => e.complete)
  const verdict = completeEvent?.verdict

  // Auto-scroll log
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight, behavior: 'smooth' })
  }, [events])

  // Get event for each stage
  const getEvent = (key) =>
    [...events].reverse().find((e) => e.step === key)

  return (
    <motion.div
      key="pipeline-drawer"
      initial={{ width: 0, opacity: 0 }}
      animate={{ width: 280, opacity: 1 }}
      exit={{ width: 0, opacity: 0 }}
      transition={{ duration: 0.28, ease: [0.4, 0, 0.2, 1] }}
      className="flex-shrink-0 border-l flex flex-col overflow-hidden"
      style={{ borderColor: 'var(--c-border)', background: 'var(--c-surface)' }}
    >
      <div className="w-[280px] h-full flex flex-col">
        {/* Header */}
        <div
          className="flex-shrink-0 flex items-center justify-between px-4 py-3 border-b"
          style={{ borderColor: 'var(--c-border)' }}
        >
          <div className="flex items-center gap-2">
            <span className="panel-label">Pipeline</span>
            {active && (
              <span
                className="w-1.5 h-1.5 rounded-full"
                style={{ background: 'var(--c-primary)', animation: 'pulse-dot 1s ease-in-out infinite' }}
              />
            )}
          </div>
          <button onClick={onClose} className="text-muted hover:text-text transition-colors">
            <X size={13} />
          </button>
        </div>

        {/* Stage visualization */}
        <div className="flex-shrink-0 p-4 border-b" style={{ borderColor: 'var(--c-border)' }}>
          {STAGES.map((stage, i) => (
            <StageNode
              key={stage.key}
              stage={stage}
              status={getStageStatus(stage.key, events, activeStep)}
              event={getEvent(stage.key)}
              isLast={i === STAGES.length - 1}
            />
          ))}

          {/* Verdict */}
          <AnimatePresence>
            {verdict && <VerdictDisplay verdict={verdict} />}
          </AnimatePresence>
        </div>

        {/* Live log */}
        <div className="flex-1 overflow-hidden flex flex-col">
          <div
            className="px-4 py-2 border-b flex items-center justify-between"
            style={{ borderColor: 'var(--c-border)' }}
          >
            <span className="panel-label">Event Log</span>
            <span className="font-mono text-[10px] text-muted">{events.length} events</span>
          </div>
          <div
            ref={logRef}
            className="flex-1 overflow-y-auto p-3 space-y-0.5"
            style={{ background: 'var(--c-bg)' }}
          >
            <AnimatePresence initial={false}>
              {events
                .filter((e) => e.step && !e.complete)
                .map((e, i) => (
                  <LogLine key={i} event={e} />
                ))}
            </AnimatePresence>

            {events.length === 0 && (
              <p className="font-mono text-[10px] text-muted italic">
                Waiting for pipeline events…
              </p>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  )
}
