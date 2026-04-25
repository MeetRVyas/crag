import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import api from '../api/client'

/* Geometric decorative lines rendered as SVG */
function GridLines() {
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" aria-hidden>
      <defs>
        <linearGradient id="lg1" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#0096D6" stopOpacity="0.3" />
          <stop offset="100%" stopColor="#0096D6" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="lg2" x1="100%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#FFD000" stopOpacity="0.2" />
          <stop offset="100%" stopColor="#FFD000" stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* Diagonal accent lines */}
      <line x1="0" y1="40%" x2="45%" y2="0" stroke="url(#lg1)" strokeWidth="1" />
      <line x1="0" y1="70%" x2="65%" y2="0" stroke="url(#lg1)" strokeWidth="0.5" />
      <line x1="100%" y1="30%" x2="55%" y2="100%" stroke="url(#lg2)" strokeWidth="1" />
      <line x1="100%" y1="60%" x2="35%" y2="100%" stroke="url(#lg2)" strokeWidth="0.5" />
      {/* Corner brackets */}
      <path d="M 60 60 L 60 80 L 80 80" stroke="#1D1F2B" strokeWidth="1" fill="none" />
      <path d="M calc(100% - 60px) 60 L calc(100% - 60px) 80 L calc(100% - 80px) 80" stroke="#1D1F2B" strokeWidth="1" fill="none" />
      <path d="M 60 calc(100% - 60px) L 60 calc(100% - 80px) L 80 calc(100% - 80px)" stroke="#1D1F2B" strokeWidth="1" fill="none" />
    </svg>
  )
}

/* Animated floating data fragments */
const FRAGMENTS = [
  'RETRIEVE', 'EVALUATE', 'REFINE', 'CORRECT', 'AMBIGUOUS', 'HyDE',
  'BM25', 'FAISS', 'RERANK', 'RAG', 'TAVILY', 'LANGGRAPH',
]

function DataFragment({ text, x, y, delay }) {
  return (
    <motion.span
      className="absolute font-mono text-[10px] text-muted pointer-events-none select-none"
      style={{ left: `${x}%`, top: `${y}%` }}
      initial={{ opacity: 0 }}
      animate={{ opacity: [0, 0.4, 0] }}
      transition={{ duration: 4, delay, repeat: Infinity, repeatDelay: Math.random() * 6 }}
    >
      {text}
    </motion.span>
  )
}

const stagger = {
  hidden: {},
  show:   { transition: { staggerChildren: 0.12 } },
}
const fadeUp = {
  hidden: { opacity: 0, y: 20 },
  show:   { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] } },
}

export default function Login() {
  const [params] = useSearchParams()
  const error    = params.get('error')
  const [hovered, setHovered] = useState(false)

  const API_BASE = import.meta.env.VITE_API_URL ?? '/api'
  const loginUrl = `${API_BASE}/auth/login`

  return (
    <div className="min-h-screen bg-bg dot-grid relative overflow-hidden flex items-center justify-center">
      {/* Atmospheric glow blobs */}
      <div
        className="absolute top-0 left-1/4 w-96 h-96 rounded-full pointer-events-none"
        style={{ background: 'radial-gradient(circle, rgba(0,150,214,0.08) 0%, transparent 70%)', filter: 'blur(40px)' }}
      />
      <div
        className="absolute bottom-1/4 right-1/4 w-64 h-64 rounded-full pointer-events-none"
        style={{ background: 'radial-gradient(circle, rgba(255,208,0,0.06) 0%, transparent 70%)', filter: 'blur(40px)' }}
      />

      {/* Scan line */}
      <div className="scan-line" />

      {/* Decorative SVG lines */}
      <GridLines />

      {/* Floating data fragments */}
      {FRAGMENTS.map((t, i) => (
        <DataFragment
          key={t}
          text={t}
          x={5 + (i * 31) % 90}
          y={8 + (i * 17) % 85}
          delay={i * 0.7}
        />
      ))}

      {/* Main card */}
      <motion.div
        className="relative z-10 w-full max-w-md mx-4"
        variants={stagger}
        initial="hidden"
        animate="show"
      >
        {/* System label */}
        <motion.div variants={fadeUp} className="flex items-center gap-3 mb-10">
          <div className="flex gap-1">
            <div className="w-2 h-2 rounded-full bg-primary" />
            <div className="w-2 h-2 rounded-full bg-yellow" />
            <div className="w-2 h-2 rounded-full bg-red" />
          </div>
          <span className="panel-label">System v1.0 · Corrective RAG</span>
        </motion.div>

        {/* Logo */}
        <motion.div variants={fadeUp}>
          <h1
            className="text-[88px] leading-none tracking-tight"
            style={{
              fontFamily: 'var(--f-display)',
              background: 'linear-gradient(135deg, var(--c-text) 0%, var(--c-dim) 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}
          >
            CRAG
          </h1>
          <div className="flex items-center gap-3 mt-1 mb-8">
            <div className="h-px flex-1 bg-gradient-to-r from-primary to-transparent" />
            <span className="font-mono text-[11px] text-dim tracking-[0.3em] uppercase">
              Corrective Retrieval‑Augmented Generation
            </span>
          </div>
        </motion.div>

        {/* Description */}
        <motion.p variants={fadeUp} className="text-dim text-sm leading-relaxed mb-8 font-body">
          Upload your documents. Ask questions. Get answers grounded in your
          content — with automatic web search fallback and a six‑stage pipeline
          that tells you exactly how confident it is.
        </motion.p>

        {/* Feature pills */}
        <motion.div variants={fadeUp} className="flex flex-wrap gap-2 mb-10">
          {['HyDE Retrieval', 'BM25 + FAISS', 'FlashRank', 'Web Fallback', 'Live Pipeline SSE'].map((f) => (
            <span
              key={f}
              className="font-mono text-[10px] px-2.5 py-1 rounded-sm text-muted"
              style={{ background: 'var(--c-surface)', border: '1px solid var(--c-border)' }}
            >
              {f}
            </span>
          ))}
        </motion.div>

        {/* Error banner */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-4 px-4 py-3 rounded-md text-red text-sm font-mono"
              style={{ background: 'var(--c-red-10)', border: '1px solid rgba(240,62,27,0.2)' }}
            >
              Authentication failed. Please try again.
            </motion.div>
          )}
        </AnimatePresence>

        {/* Google login button */}
        <motion.div variants={fadeUp}>
          <a
            href={loginUrl}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
            className="flex items-center justify-center gap-3 w-full py-4 rounded-md relative overflow-hidden font-condensed font-semibold tracking-widest text-sm uppercase text-white transition-all duration-300"
            style={{
              background: hovered
                ? 'linear-gradient(135deg, #00AAEE 0%, var(--c-primary) 100%)'
                : 'linear-gradient(135deg, var(--c-primary) 0%, #006DA0 100%)',
              boxShadow: hovered
                ? '0 0 32px rgba(0,150,214,0.4), 0 4px 24px rgba(0,0,0,0.4)'
                : '0 0 12px rgba(0,150,214,0.15), 0 4px 12px rgba(0,0,0,0.3)',
              letterSpacing: '0.15em',
            }}
          >
            {/* Google icon */}
            <svg width="18" height="18" viewBox="0 0 48 48" fill="none">
              <path d="M43.611 20.083H42V20H24v8h11.303c-1.649 4.657-6.08 8-11.303 8-6.627 0-12-5.373-12-12s5.373-12 12-12c3.059 0 5.842 1.154 7.961 3.039l5.657-5.657C34.046 6.053 29.268 4 24 4 12.955 4 4 12.955 4 24s8.955 20 20 20 20-8.955 20-20c0-1.341-.138-2.65-.389-3.917z" fill="#FFF" fillOpacity="0.9"/>
            </svg>
            Continue with Google
            <motion.div
              className="absolute inset-0 bg-white"
              initial={{ opacity: 0 }}
              animate={{ opacity: hovered ? 0.04 : 0 }}
            />
          </a>
        </motion.div>

        {/* Footer note */}
        <motion.p variants={fadeUp} className="mt-6 text-center text-[11px] text-muted font-mono">
          Authentication is handled via Google OAuth 2.0 · Session expires in 24h
        </motion.p>
      </motion.div>
    </div>
  )
}
