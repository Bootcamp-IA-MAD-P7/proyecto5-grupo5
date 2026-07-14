import { useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { LoaderCircle } from 'lucide-react'
import FormSection from '@/components/FormSection'
import PredictionResult from '@/components/PredictionResult'
import ThemeToggle from '@/components/ThemeToggle'
import LanguageToggle from '@/components/LanguageToggle'
import { GROUPS } from '@/data/fields'
import { usePredictForm } from '@/hooks/usePredictForm'
import { useLanguage } from '@/i18n'
import { logger } from '@/lib/logger'

export default function PredictPage() {
  const { formData, loading, result, error, handleChange, handleSubmit } = usePredictForm()
  const { t, lang } = useLanguage()

  useEffect(() => {
    logger.log(`PredictPage mounted (lang=${lang})`)
    return () => logger.log('PredictPage unmounted')
  }, [lang])

  useEffect(() => {
    if (error) logger.warn('Error state updated', error)
  }, [error])

  return (
    <div className="min-h-screen bg-muted/30 p-4 md:p-8">
      <div className="mx-auto max-w-4xl space-y-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">{t.page.title}</h1>
            <p className="text-muted-foreground mt-1">{t.page.description}</p>
          </div>
          <div className="flex gap-2">
            <LanguageToggle />
            <ThemeToggle />
          </div>
        </div>

        <Button type="submit" size="lg" className="w-full" disabled={loading} onClick={handleSubmit}>
          {loading && <LoaderCircle className="h-5 w-5 animate-spin" />}
          {loading ? t.buttons.predicting : t.buttons.predict}
        </Button>

        {error && (
          <Card className="border-destructive">
            <CardContent className="pt-6">
              <p className="text-destructive font-medium">{t.error} {error}</p>
            </CardContent>
          </Card>
        )}

        {result && (
          <PredictionResult
            prediction={result.prediction}
            probability={result.probability}
          />
        )}

        <form className="space-y-6">
          {GROUPS.map((group) => (
            <FormSection
              key={group}
              group={group}
              values={formData}
              onChange={handleChange}
            />
          ))}
        </form>
      </div>
    </div>
  )
}
