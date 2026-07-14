import { useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { useLanguage } from '@/i18n'
import { logger } from '@/lib/logger'

interface PredictionResultProps {
  prediction: number
  probability: number
}

const COLORS = {
  low: { bar: 'bg-emerald-500', text: 'text-emerald-500' },
  medium: { bar: 'bg-amber-500', text: 'text-amber-500' },
  high: { bar: 'bg-rose-500', text: 'text-rose-500' },
} as const

function getRisk(p: number) {
  if (p <= 0.33) return { level: 'low' as keyof typeof COLORS, color: COLORS.low }
  if (p <= 0.66) return { level: 'medium' as keyof typeof COLORS, color: COLORS.medium }
  return { level: 'high' as keyof typeof COLORS, color: COLORS.high }
}

export default function PredictionResult({ prediction, probability }: PredictionResultProps) {
  const { t } = useLanguage()
  const pct = (probability * 100).toFixed(1)
  const risk = getRisk(probability)
  const answer = prediction === 1 ? t.result.yes : t.result.no

  useEffect(() => {
    logger.log('PredictionResult mounted', { prediction, probability })
  }, [prediction, probability])

  return (
    <Card>
      <CardHeader className="text-center">
        <CardTitle>{t.result.title}</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col items-center gap-4">
        <p className="text-base">
          {t.result.sentence.start}
          <span className="font-bold">{answer}</span>
          {t.result.sentence.end}
        </p>

        <div className={`text-sm font-semibold ${risk.color.text}`}>
          {t.result.riskLabels[risk.level]}
        </div>

        <div className="w-full max-w-sm space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">{t.result.probability}</span>
            <span className="font-semibold">{pct}%</span>
          </div>
          <div className="h-3 w-full rounded-full bg-muted overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${risk.color.bar}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
