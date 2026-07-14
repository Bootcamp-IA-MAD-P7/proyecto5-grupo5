export type Language = 'es' | 'en'

export interface Translations {
  page: {
    title: string
    description: string
  }
  buttons: {
    predict: string
    predicting: string
    home: string
    themeLight: string
    themeDark: string
    langEs: string
    langEn: string
  }
  groups: Record<string, string>
  fields: Record<string, {
    label: string
    options?: Record<string, string>
  }>
  result: {
    title: string
    sentence: { start: string; end: string }
    probability: string
    risk: string
    riskLabels: Record<'low' | 'medium' | 'high', string>
    yes: string
    no: string
  }
  error: string
  placeholder: string
}
