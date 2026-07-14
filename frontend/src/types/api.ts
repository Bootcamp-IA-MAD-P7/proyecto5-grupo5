export interface PredictPayload {
  Gender: string
  'Senior Citizen': number
  Partner: string
  Dependents: string
  'Tenure Months': number
  'Phone Service': string
  'Multiple Lines': string
  'Internet Service': string
  'Online Security': string
  'Online Backup': string
  'Device Protection': string
  'Tech Support': string
  'Streaming TV': string
  'Streaming Movies': string
  Contract: string
  'Paperless Billing': string
  'Payment Method': string
  'Monthly Charges': number
  'Total Charges': number
}

export interface PredictResponse {
  prediction: number
  probability: number
}
