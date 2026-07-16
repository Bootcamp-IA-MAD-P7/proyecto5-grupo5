import { useState } from 'react'
import { predict } from '@/api/predict'
import type { PredictPayload } from '@/types/api'
import { buildInitialState } from '@/data/fields'
import { logger } from '@/lib/logger'

export function usePredictForm() {
  const [formData, setFormData] = useState<Record<string, string>>(() => {
    const initial = buildInitialState()
    logger.log('Form initialized with defaults', initial)
    return initial
  })
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{ prediction: number; probability: number } | null>(null)
  const [error, setError] = useState<string | null>(null)

  function handleChange(name: string, value: string) {
    logger.log(`Field changed: ${name} => "${value}"`)
    setFormData((prev) => ({ ...prev, [name]: value }))
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    logger.log('Form submitted')

    setLoading(true)
    setError(null)
    setResult(null)

    const payload: PredictPayload = {
      Gender: formData['Gender'],
      'Senior Citizen': Number(formData['Senior Citizen']),
      Partner: formData['Partner'],
      Dependents: formData['Dependents'],
      'Tenure Months': Number(formData['Tenure Months']),
      'Phone Service': formData['Phone Service'],
      'Multiple Lines': formData['Multiple Lines'],
      'Internet Service': formData['Internet Service'],
      'Online Security': formData['Online Security'],
      'Online Backup': formData['Online Backup'],
      'Device Protection': formData['Device Protection'],
      'Tech Support': formData['Tech Support'],
      'Streaming TV': formData['Streaming TV'],
      'Streaming Movies': formData['Streaming Movies'],
      Contract: formData['Contract'],
      'Paperless Billing': formData['Paperless Billing'],
      'Payment Method': formData['Payment Method'],
      'Monthly Charges': Number(formData['Monthly Charges']),
      'Total Charges': Number(formData['Total Charges']),
    }

    logger.log('Payload built', payload)

    try {
      const res = await predict(payload)
      logger.log('Prediction successful', res)
      setResult(res)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error'
      logger.error('Prediction failed', message)
      setError(message)
    } finally {
      setLoading(false)
    }
  }

  return { formData, loading, result, error, handleChange, handleSubmit }
}
