import { useState, useRef, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, FileText, Trash2, Zap, CheckCircle2,
  AlertCircle, Clock, RefreshCw, ChevronDown, ChevronRight,
  Layers
} from 'lucide-react'
import api from '../api/client'

/* ── Tiny helpers ─────────────────────────────────────── */
function StatusDot({ status }) {
  const map = {
    ready:      { color: 'var(--c-green)',   pulse: false },
    uploading:  { color: 'var(--c-yellow)',  pulse: true  },
    processing: { color: 'var(--c-primary)', pulse: true  },
    error:      { color: 'var(--c-red)',     pulse: false },
    pending:    { color: 'var(--c-muted)',   pulse: false },
  }
  const { color, pulse } = map[status] ?? map.pending
  return (
    <span
      className="inline-block w-1.5 h-1.5 rounded-full flex-shrink-0"
      style={{
        background: color,
        animation: pulse ? 'pulse-dot 1.2s ease-in-out infinite' : 'none',
      }}
    />
  )
}

function FileRow({ file, onDelete, disabled }) {
  const [deleting, setDeleting] = useState(false)
  const name = file.filename

  const handleDelete = async () => {
    if (deleting || disabled) return
    setDeleting(true)
    try { await api.documents.delete(name) }
    catch (_) {}
    finally { onDelete(name) }
  }

  return (
    <motion.div
      layout
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -8 }}
      className="group flex items-center gap-2 px-3 py-2 rounded hover:bg-surface-hi transition-colors"
    >
      <FileText size={12} className="flex-shrink-0 text-primary" />
      <span
        className="flex-1 truncate font-mono text-[11px] text-dim"
        title={name}
      >
        {name}
      </span>
      <button
        onClick={handleDelete}
        className="opacity-0 group-hover:opacity-100 transition-opacity text-muted hover:text-red"
        title="Delete"
      >
        {deleting ? <RefreshCw size={11} className="animate-spin" /> : <Trash2 size={11} />}
      </button>
    </motion.div>
  )
}

/* ── Drop Zone ────────────────────────────────────────── */
function DropZone({ onUpload, disabled }) {
  const inputRef      = useRef()
  const [dragging, setDragging] = useState(false)
  const [progress, setProgress] = useState(null) // 0-100

  const upload = useCallback(async (file) => {
    if (!file || !file.name.endsWith('.pdf')) return
    setProgress(0)
    try {
      await api.documents.upload(file, setProgress)
      onUpload()
    } catch (err) {
      console.error(err)
    } finally {
      setProgress(null)
    }
  }, [onUpload])

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    upload(file)
  }

  return (
    <div
      className={`relative border border-dashed rounded-md p-4 text-center cursor-pointer transition-all duration-200 ${
        dragging ? 'drop-active' : 'border-border hover:border-border-hi'
      }`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={(e) => upload(e.target.files[0])}
        disabled={disabled}
      />
      <Upload size={16} className="mx-auto mb-2 text-muted" />
      <p className="font-mono text-[11px] text-dim">
        {progress !== null
          ? `Uploading… ${progress}%`
          : 'Drop PDF or click to upload'}
      </p>
      {progress !== null && (
        <div className="mt-2 h-0.5 bg-border rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-primary"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
          />
        </div>
      )}
    </div>
  )
}

/* ── Snapshot item ────────────────────────────────────── */
function SnapshotItem({ snap, index, total }) {
  const [open, setOpen] = useState(false)
  const isLatest = index === total - 1
  const dt = new Date(snap.created_at)
  const label = `${dt.toLocaleDateString()} ${dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`

  return (
    <div className="relative">
      {/* Timeline line */}
      {index < total - 1 && (
        <div
          className="absolute left-[9px] top-6 bottom-0 w-px"
          style={{ background: 'var(--c-border)' }}
        />
      )}
      <div className="flex items-start gap-2.5">
        {/* Node */}
        <div
          className="flex-shrink-0 w-[18px] h-[18px] rounded-full flex items-center justify-center mt-0.5"
          style={{
            background: isLatest ? 'var(--c-primary)' : 'var(--c-surface-hi)',
            border: `1px solid ${isLatest ? 'var(--c-primary)' : 'var(--c-border)'}`,
            boxShadow: isLatest ? '0 0 8px var(--c-primary-20)' : 'none',
          }}
        >
          {isLatest && <div className="w-1.5 h-1.5 rounded-full bg-white" />}
        </div>
        {/* Content */}
        <div className="flex-1 min-w-0 pb-3">
          <button
            onClick={() => setOpen((v) => !v)}
            className="flex items-center gap-1 w-full text-left"
          >
            <span className="font-mono text-[10px] text-dim truncate">{snap.id}</span>
            {isLatest && (
              <span
                className="text-[9px] px-1.5 py-0.5 rounded-sm flex-shrink-0"
                style={{ background: 'var(--c-primary-10)', color: 'var(--c-primary)', fontFamily: 'var(--f-mono)' }}
              >
                CURRENT
              </span>
            )}
            <div className="flex-1" />
            {open ? <ChevronDown size={10} className="text-muted" /> : <ChevronRight size={10} className="text-muted" />}
          </button>
          <p className="font-mono text-[9px] text-muted mt-0.5">{label}</p>
          <AnimatePresence>
            {open && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="mt-1.5 space-y-0.5">
                  {snap.files.map((f) => (
                    <p key={f.filename} className="font-mono text-[9px] text-muted pl-1 truncate">
                      · {f.filename}
                    </p>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}

/* ── Document Panel ───────────────────────────────────── */
export default function DocumentPanel({ retrieverConfig }) {
  const [files,      setFiles]      = useState([])
  const [snapshots,  setSnapshots]  = useState([])
  const [processing, setProcessing] = useState(false)
  const [indexed,    setIndexed]    = useState(false)
  const [showSnaps,  setShowSnaps]  = useState(false)
  const [error,      setError]      = useState(null)

  const loadFiles = useCallback(async () => {
    try {
      const [fRes, sRes] = await Promise.all([
        api.documents.list(),
        api.documents.snapshots(),
      ])
      setFiles(fRes.data.documents ?? [])
      setSnapshots(sRes.data.snapshots ?? [])
    } catch (e) {
      console.error(e)
    }
  }, [])

  useEffect(() => { loadFiles() }, [loadFiles])

  const handleUpload = useCallback(async () => {
    setIndexed(false)
    await loadFiles()
  }, [loadFiles])

  const handleDelete = useCallback((name) => {
    setFiles((prev) => prev.filter((f) => f.filename !== name))
    setIndexed(false)
  }, [])

  const handleProcess = async () => {
    setProcessing(true)
    setError(null)
    try {
      await api.documents.process(
        retrieverConfig.embProvider,
        retrieverConfig.embModel,
      )
      setIndexed(true)
      await loadFiles()
    } catch (e) {
      setError(e.response?.data?.detail ?? 'Processing failed')
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Panel header */}
      <div
        className="flex-shrink-0 px-4 py-3 border-b flex items-center justify-between"
        style={{ borderColor: 'var(--c-border)' }}
      >
        <span className="panel-label">Documents</span>
        <span
          className="font-mono text-[10px] px-2 py-0.5 rounded-sm"
          style={{
            background: indexed ? 'var(--c-green-10)' : 'var(--c-surface-hi)',
            color: indexed ? 'var(--c-green)' : 'var(--c-muted)',
          }}
        >
          {indexed ? 'INDEXED' : files.length === 0 ? 'EMPTY' : 'NOT INDEXED'}
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* Drop zone */}
        <DropZone onUpload={handleUpload} disabled={processing} />

        {/* File list */}
        {files.length > 0 && (
          <div>
            <p className="panel-label mb-2 px-1">
              {files.length} file{files.length !== 1 ? 's' : ''}
            </p>
            <AnimatePresence>
              {files.map((f) => (
                <FileRow
                  key={f.filename}
                  file={f}
                  onDelete={handleDelete}
                  disabled={processing}
                />
              ))}
            </AnimatePresence>
          </div>
        )}

        {/* Process button */}
        {files.length > 0 && (
          <div>
            <button
              onClick={handleProcess}
              disabled={processing}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-md font-condensed font-semibold text-sm tracking-widest uppercase transition-all duration-200"
              style={{
                background: processing
                  ? 'var(--c-surface-hi)'
                  : 'linear-gradient(135deg, var(--c-primary) 0%, #006DA0 100%)',
                color: processing ? 'var(--c-muted)' : 'white',
                boxShadow: processing ? 'none' : '0 0 16px var(--c-primary-20)',
                letterSpacing: '0.12em',
              }}
            >
              {processing ? (
                <>
                  <RefreshCw size={13} className="animate-spin" />
                  Indexing…
                </>
              ) : (
                <>
                  <Zap size={13} />
                  {indexed ? 'Re-Index' : 'Build Index'}
                </>
              )}
            </button>
            {error && (
              <p className="mt-2 text-[11px] text-red font-mono flex items-center gap-1">
                <AlertCircle size={10} />
                {error}
              </p>
            )}
          </div>
        )}

        {/* Snapshot timeline */}
        {snapshots.length > 0 && (
          <div>
            <button
              onClick={() => setShowSnaps((v) => !v)}
              className="flex items-center gap-2 w-full mb-2"
            >
              <Layers size={11} className="text-muted" />
              <span className="panel-label flex-1 text-left">Snapshots ({snapshots.length})</span>
              {showSnaps ? <ChevronDown size={10} className="text-muted" /> : <ChevronRight size={10} className="text-muted" />}
            </button>
            <AnimatePresence>
              {showSnaps && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  className="overflow-hidden pl-1"
                >
                  {[...snapshots].reverse().map((s, i, arr) => (
                    <SnapshotItem
                      key={s.id}
                      snap={s}
                      index={arr.length - 1 - i}
                      total={arr.length}
                    />
                  ))}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        )}
      </div>
    </div>
  )
}
