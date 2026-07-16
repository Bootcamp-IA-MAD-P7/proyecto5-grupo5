import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider } from '@/context/ThemeContext'
import { LanguageProvider } from '@/i18n'
import './index.css'
import App from './App.tsx'
import { logger } from '@/lib/logger'

const env = import.meta.env.VITE_ENV || 'dev'
logger.log(`App starting (ENV=${env})`)

createRoot(document.getElementById('root')!).render(
    <BrowserRouter>
      <LanguageProvider>
        <ThemeProvider>
          <App />
        </ThemeProvider>
      </LanguageProvider>
    </BrowserRouter>
)
