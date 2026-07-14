import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { FieldDef } from '@/types/ui'
import { useLanguage } from '@/i18n'
import { logger } from '@/lib/logger'

interface FormFieldProps {
  field: FieldDef
  value: string
  onChange: (name: string, value: string) => void
}

export default function FormField({ field, value, onChange }: FormFieldProps) {
  const { t } = useLanguage()
  const fieldT = t.fields[field.name]
  const label = fieldT?.label ?? field.name

  logger.log(`Rendering FormField: ${field.name} = "${value}"`)

  return (
    <div className="space-y-1.5">
      <Label htmlFor={field.name}>{label}</Label>
      {field.type === 'select' && field.options.length > 0 ? (
        <Select
          value={value}
          onValueChange={(v) => {
            logger.log(`Select onChange: ${field.name} -> "${v}"`)
            onChange(field.name, v)
          }}
        >
          <SelectTrigger id={field.name}>
            <SelectValue placeholder={t.placeholder} />
          </SelectTrigger>
          <SelectContent>
            {field.options.map((opt) => (
              <SelectItem key={opt} value={opt}>
                {fieldT?.options?.[opt] ?? opt}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : (
        <Input
          id={field.name}
          type="number"
          step={field.step || '1'}
          value={value}
          onChange={(e) => {
            logger.log(`Input onChange: ${field.name} -> "${e.target.value}"`)
            onChange(field.name, e.target.value)
          }}
          placeholder="0"
        />
      )}
    </div>
  )
}
