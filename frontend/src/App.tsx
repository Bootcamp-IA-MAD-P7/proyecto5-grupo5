import { useEffect } from 'react'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import ThemeToggle from '@/components/ThemeToggle'
import LanguageToggle from '@/components/LanguageToggle'
import PredictPage from '@/pages/PredictPage'
import { useLanguage } from '@/i18n'
import { logger } from '@/lib/logger'

function Home() {
  const { t } = useLanguage()

  useEffect(() => {
    logger.log('Home page mounted')
  }, [])

  return (
    <div className="min-h-screen bg-muted/30 flex flex-col items-center justify-center p-4">
      <div className="absolute top-4 right-4 flex gap-2">
        <LanguageToggle />
        <ThemeToggle />
      </div>
      <h1 className="text-4xl font-bold mb-4">{t.page.title}</h1>
      <p className="text-muted-foreground mb-8 text-center max-w-md">
        {t.page.description}
      </p>
      <Link to="/predict">
        <Button variant="default" size="lg">
          {t.buttons.predict}
        </Button>
      </Link>
    </div>
  )
}

function App() {
  const location = useLocation()

  useEffect(() => {
    logger.log(`Route changed: ${location.pathname}`)
  }, [location])

  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/predict" element={<PredictPage />} />
    </Routes>
  )
}

export default App
