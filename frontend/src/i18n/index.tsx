import { createContext, useContext, useState, useEffect, useCallback, type ReactNode } from 'react'
import type { Language, Translations } from './types'
import { es } from './es'
import { en } from './en'
import { logger } from '@/lib/logger'

interface LanguageContextType {
  lang: Language
  t: Translations
  toggleLang: () => void
}

const translations: Record<Language, Translations> = { es, en }

const LanguageContext = createContext<LanguageContextType | null>(null)

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Language>('es')

  useEffect(() => {
    document.documentElement.lang = lang
    logger.log(`Language set to ${lang}`)
  }, [lang])

  const toggleLang = useCallback(() => {
    setLang((prev) => (prev === 'es' ? 'en' : 'es'))
  }, [])

  return (
    <LanguageContext.Provider value={{ lang, t: translations[lang], toggleLang }}>
      {children}
    </LanguageContext.Provider>
  )
}

export function useLanguage() {
  const ctx = useContext(LanguageContext)
  if (!ctx) throw new Error('useLanguage must be used within LanguageProvider')
  return ctx
}
