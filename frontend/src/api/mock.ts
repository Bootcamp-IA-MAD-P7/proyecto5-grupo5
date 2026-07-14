import type { PredictPayload, PredictResponse } from '@/types/api'
import { logger } from '@/lib/logger'

export function mockPredict(_data: PredictPayload): Promise<PredictResponse> {
  const probability = Math.random()
  const churn = probability > 0.5 ? 1 : 0

  logger.log('Mock predict returning', { prediction: churn, probability })
  return new Promise((resolve) =>
    setTimeout(() => resolve({ prediction: churn, probability }), 1500),
  )
}
