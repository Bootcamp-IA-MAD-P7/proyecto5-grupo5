import { Button } from '@/components/ui/button'
import { useLanguage } from '@/i18n'
import { Languages } from 'lucide-react'

export default function LanguageToggle() {
  const { lang, toggleLang, t } = useLanguage()

  return (
    <Button
      variant="outline"
      size="icon"
      onClick={toggleLang}
      title={lang === 'es' ? t.buttons.langEn : t.buttons.langEs}
    >
      <Languages className="h-4 w-4" />
    </Button>
  )
}
