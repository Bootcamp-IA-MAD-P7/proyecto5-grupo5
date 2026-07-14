import type { PredictPayload, PredictResponse } from '@/types/api'
import api from '@/api/client'
import { mockPredict } from '@/api/mock'
import { logger } from '@/lib/logger'

export async function predict(data: PredictPayload): Promise<PredictResponse> {
  logger.log('Calling predict() with payload', data)

  if (import.meta.env.VITE_MOCK === 'true') {
    return mockPredict(data)
  }

  const res = await api.post<PredictResponse>('/predict', data)
  logger.log('Predict result', res.data)
  return res.data
}
