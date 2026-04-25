import { createContext, useContext, useState, useEffect, useCallback } from 'react'
import api from '../api/client'

const AuthCtx = createContext(null)

export function AuthProvider({ children }) {
  const [token, setToken]   = useState(() => localStorage.getItem('crag_token'))
  const [loading, setLoading] = useState(false)

  const login = useCallback((jwt) => {
    localStorage.setItem('crag_token', jwt)
    setToken(jwt)
  }, [])

  const logout = useCallback(async () => {
    try { await api.auth.logout() } catch (_) {}
    localStorage.removeItem('crag_token')
    setToken(null)
  }, [])

  // Listen for forced logout from axios interceptor
  useEffect(() => {
    const handler = () => {
      localStorage.removeItem('crag_token')
      setToken(null)
    }
    window.addEventListener('crag:logout', handler)
    return () => window.removeEventListener('crag:logout', handler)
  }, [])

  return (
    <AuthCtx.Provider value={{ token, isAuthed: !!token, login, logout, loading }}>
      {children}
    </AuthCtx.Provider>
  )
}

export const useAuth = () => useContext(AuthCtx)
