import { useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { useAuth } from '../context/AuthContext'
import { motion } from 'framer-motion'

/**
 * The backend's /auth/callback returns JSON { access_token, token_type }.
 * To bridge SPA ↔ backend OAuth, the backend should redirect to
 *   http://frontend-url/callback?token=<jwt>
 *
 * If you need to modify the backend to do this, add to auth router callback:
 *   return RedirectResponse(f"{settings.FRONTEND_URL}/callback?token={token}")
 *
 * This page handles extracting the token from the URL param.
 */
export default function Callback() {
  const [params]   = useSearchParams()
  const { login }  = useAuth()
  const navigate   = useNavigate()

  useEffect(() => {
    const token = params.get('token') || params.get('access_token')
    if (token) {
      login(token)
      navigate('/dashboard', { replace: true })
    } else {
      // If no token found, redirect to login with error
      navigate('/login?error=oauth_failed', { replace: true })
    }
  }, [params, login, navigate])

  return (
    <div className="min-h-screen bg-bg dot-grid flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="text-center"
      >
        {/* Orbital loader */}
        <div className="relative w-16 h-16 mx-auto mb-6">
          <div className="absolute inset-0 rounded-full border border-border" />
          <div
            className="absolute top-1/2 left-1/2 w-2.5 h-2.5 rounded-full bg-primary -mt-[5px] -ml-[5px]"
            style={{ animation: 'orbit 1.2s linear infinite' }}
          />
        </div>
        <p className="font-mono text-dim text-xs tracking-widest uppercase">Authenticating…</p>
      </motion.div>
    </div>
  )
}
