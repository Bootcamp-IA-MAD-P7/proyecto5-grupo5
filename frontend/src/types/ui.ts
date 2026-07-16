import type { PredictPayload } from '@/types/api'

export interface FieldDef {
  name: keyof PredictPayload
  type: 'select' | 'number'
  options: string[]
  step?: string
  group: string
  defaultValue: string
}
