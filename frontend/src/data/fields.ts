import type { FieldDef } from '@/types/ui'

export const FIELDS: FieldDef[] = [
  { name: 'Gender', type: 'select', options: ['Male', 'Female'], group: 'personal-info', defaultValue: 'Male' },
  { name: 'Senior Citizen', type: 'select', options: ['0', '1'], group: 'personal-info', defaultValue: '0' },
  { name: 'Partner', type: 'select', options: ['No', 'Yes'], group: 'personal-info', defaultValue: 'No' },
  { name: 'Dependents', type: 'select', options: ['No', 'Yes'], group: 'personal-info', defaultValue: 'No' },
  { name: 'Tenure Months', type: 'number', options: [], group: 'account-info', defaultValue: '1' },
  { name: 'Phone Service', type: 'select', options: ['No', 'Yes'], group: 'services', defaultValue: 'Yes' },
  { name: 'Multiple Lines', type: 'select', options: ['No', 'Yes'], group: 'services', defaultValue: 'No' },
  { name: 'Internet Service', type: 'select', options: ['DSL', 'Fiber optic', 'No'], group: 'services', defaultValue: 'Fiber optic' },
  { name: 'Online Security', type: 'select', options: ['No', 'Yes'], group: 'services', defaultValue: 'No' },
  { name: 'Online Backup', type: 'select', options: ['No', 'Yes'], group: 'services', defaultValue: 'No' },
  { name: 'Device Protection', type: 'select', options: ['No', 'Yes'], group: 'services', defaultValue: 'No' },
  { name: 'Tech Support', type: 'select', options: ['No', 'Yes'], group: 'services', defaultValue: 'No' },
  { name: 'Streaming TV', type: 'select', options: ['No', 'Yes'], group: 'services', defaultValue: 'No' },
  { name: 'Streaming Movies', type: 'select', options: ['No', 'Yes'], group: 'services', defaultValue: 'No' },
  { name: 'Contract', type: 'select', options: ['Month-to-month', 'One year', 'Two year'], group: 'contract-billing', defaultValue: 'Month-to-month' },
  { name: 'Paperless Billing', type: 'select', options: ['No', 'Yes'], group: 'contract-billing', defaultValue: 'Yes' },
  { name: 'Payment Method', type: 'select', options: ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], group: 'contract-billing', defaultValue: 'Electronic check' },
  { name: 'Monthly Charges', type: 'number', options: [], step: '0.01', group: 'contract-billing', defaultValue: '20.05' },
  { name: 'Total Charges', type: 'number', options: [], step: '0.01', group: 'contract-billing', defaultValue: '20.2' },
]

export const GROUPS = [...new Set(FIELDS.map((f) => f.group))]

export function buildInitialState(): Record<string, string> {
  const state: Record<string, string> = {}
  FIELDS.forEach((f) => {
    state[f.name] = f.defaultValue
  })
  return state
}
