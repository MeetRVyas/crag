import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { LogOut, Key, Cpu, ChevronLeft, ChevronRight } from 'lucide-react'
import { useAuth } from '../context/AuthContext'
import DocumentPanel   from './DocumentPanel'
import ChatPanel       from './ChatPanel'
import PipelineDrawer  from './PipelineDrawer'
import ModelManager    from './ModelManager'
import ApiKeyModal     from './ApiKeyModal'

/* ── Header ──────────────────────────────────────────────────────────────── */
function Header({ onOpenModels, onOpenKeys, sidebarOpen, setSidebarOpen }) {
  const { logout } = useAuth()

  return (
    <header
      className="flex-shrink-0 flex items-center gap-4 px-5 border-b"
      style={{
        height: 52,
        borderColor: 'var(--c-border)',
        background: 'rgba(8,9,14,0.92)',
        backdropFilter: 'blur(8px)',
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-2.5 mr-2">
        <div className="flex gap-[3px]">
          <div className="w-[5px] h-4 rounded-sm bg-primary" />
          <div className="w-[5px] h-4 rounded-sm bg-yellow" style={{ marginTop: 3 }} />
          <div className="w-[5px] h-4 rounded-sm bg-red" style={{ marginTop: 6 }} />
        </div>
        <span
          className="text-2xl tracking-wide text-text select-none"
          style={{ fontFamily: 'var(--f-display)', letterSpacing: '0.12em' }}
        >
          CRAG
        </span>
      </div>

      {/* Sidebar toggle */}
      <button
        onClick={() => setSidebarOpen((v) => !v)}
        className="btn-ghost p-1.5"
        title={sidebarOpen ? 'Collapse documents' : 'Expand documents'}
      >
        {sidebarOpen ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
      </button>

      <div className="flex-1" />

      {/* Actions */}
      <button onClick={onOpenModels} className="btn-ghost flex items-center gap-1.5">
        <Cpu size={12} />
        Models
      </button>
      <button onClick={onOpenKeys} className="btn-ghost flex items-center gap-1.5">
        <Key size={12} />
        API Keys
      </button>
      <div className="w-px h-5 bg-border mx-1" />
      <button
        onClick={logout}
        className="btn-ghost flex items-center gap-1.5 text-red border-transparent hover:border-red hover:bg-red-10"
        style={{ '--tw-border-opacity': 1 }}
        title="Logout"
      >
        <LogOut size={12} />
        Logout
      </button>
    </header>
  )
}

/* ── Dashboard ───────────────────────────────────────────────────────────── */
export default function Dashboard() {
  const [sidebarOpen,  setSidebarOpen]  = useState(true)
  const [showModels,   setShowModels]   = useState(false)
  const [showKeys,     setShowKeys]     = useState(false)

  // Pipeline state — lifted so header and chat can share it
  const [pipelineEvents, setPipelineEvents] = useState([])   // [{step, message, timestamp}]
  const [pipelineActive, setPipelineActive] = useState(false)

  // Current retrieval config (passed from chat to document panel and vice-versa)
  const [retrieverConfig, setRetrieverConfig] = useState({
    llmProvider: 'ollama',
    llmModel:    '',
    embProvider: 'ollama',
    embModel:    '',
  })

  const handlePipelineEvent = useCallback((event) => {
    setPipelineEvents((prev) => [...prev, event])
    if (event.complete) setPipelineActive(false)
  }, [])

  const startPipeline = useCallback(() => {
    setPipelineEvents([])
    setPipelineActive(true)
  }, [])

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-bg">
      {/* Ambient background glow */}
      <div
        className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] pointer-events-none"
        style={{ background: 'radial-gradient(ellipse, rgba(0,150,214,0.04) 0%, transparent 70%)', filter: 'blur(60px)' }}
      />

      <Header
        onOpenModels={() => setShowModels(true)}
        onOpenKeys={() => setShowKeys(true)}
        sidebarOpen={sidebarOpen}
        setSidebarOpen={setSidebarOpen}
      />

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left: Document panel */}
        <AnimatePresence initial={false}>
          {sidebarOpen && (
            <motion.div
              key="sidebar"
              initial={{ width: 0, opacity: 0 }}
              animate={{ width: 280, opacity: 1 }}
              exit={{ width: 0, opacity: 0 }}
              transition={{ duration: 0.28, ease: [0.4, 0, 0.2, 1] }}
              className="flex-shrink-0 overflow-hidden border-r"
              style={{ borderColor: 'var(--c-border)' }}
            >
              <div className="w-[280px] h-full">
                <DocumentPanel retrieverConfig={retrieverConfig} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Center: Chat */}
        <div className="flex-1 overflow-hidden">
          <ChatPanel
            retrieverConfig={retrieverConfig}
            setRetrieverConfig={setRetrieverConfig}
            onPipelineStart={startPipeline}
            onPipelineEvent={handlePipelineEvent}
            pipelineActive={pipelineActive}
          />
        </div>

        {/* Right: Pipeline drawer */}
        <AnimatePresence>
          {(pipelineActive || pipelineEvents.length > 0) && (
            <PipelineDrawer
              events={pipelineEvents}
              active={pipelineActive}
              onClose={() => { setPipelineEvents([]); setPipelineActive(false) }}
            />
          )}
        </AnimatePresence>
      </div>

      {/* Modals */}
      <AnimatePresence>
        {showModels && <ModelManager onClose={() => setShowModels(false)} />}
        {showKeys   && <ApiKeyModal  onClose={() => setShowKeys(false)} />}
      </AnimatePresence>
    </div>
  )
}
