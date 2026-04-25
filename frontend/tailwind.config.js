/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Bebas Neue"', 'sans-serif'],
        body:    ['Barlow', 'sans-serif'],
        mono:    ['"IBM Plex Mono"', 'monospace'],
      },
      colors: {
        bg:          '#08090E',
        surface:     '#111218',
        'surface-hi':'#181A22',
        border:      '#1D1F2B',
        'border-hi': '#282C3E',
        primary:     '#0096D6',
        yellow:      '#FFD000',
        red:         '#F03E1B',
        text:        '#E8EAF2',
        dim:         '#7880A0',
        muted:       '#40455A',
      },
      keyframes: {
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(12px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        shimmer: {
          '0%':   { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition:  '200% 0' },
        },
        'pulse-dot': {
          '0%, 100%': { opacity: '1', transform: 'scale(1)' },
          '50%':      { opacity: '0.4', transform: 'scale(0.8)' },
        },
        'scan-line': {
          '0%':   { transform: 'translateY(-100%)', opacity: '0' },
          '10%':  { opacity: '0.6' },
          '90%':  { opacity: '0.6' },
          '100%': { transform: 'translateY(100vh)', opacity: '0' },
        },
        orbit: {
          '0%':   { transform: 'rotate(0deg) translateX(28px) rotate(0deg)' },
          '100%': { transform: 'rotate(360deg) translateX(28px) rotate(-360deg)' },
        },
      },
      animation: {
        'fade-up':   'fade-up 0.5s ease forwards',
        shimmer:     'shimmer 2.5s linear infinite',
        'pulse-dot': 'pulse-dot 1.4s ease-in-out infinite',
        'scan-line': 'scan-line 4s linear infinite',
        orbit:       'orbit 3s linear infinite',
      },
    },
  },
  plugins: [],
}
