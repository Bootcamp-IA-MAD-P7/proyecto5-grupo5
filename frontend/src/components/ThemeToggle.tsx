import { Button } from '@/components/ui/button'
import { useTheme } from '@/context/ThemeContext'
import { useLanguage } from '@/i18n'
import { Sun, Moon } from 'lucide-react'

export default function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  const { t } = useLanguage()

  return (
    <Button
      variant="outline"
      size="icon"
      onClick={toggleTheme}
      title={theme === 'light' ? t.buttons.themeDark : t.buttons.themeLight}
    >
      {theme === 'light' ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
    </Button>
  )
}
