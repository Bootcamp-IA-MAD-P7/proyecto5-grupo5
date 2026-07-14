import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import FormField from '@/components/FormField'
import { FIELDS } from '@/data/fields'
import { useLanguage } from '@/i18n'
import { logger } from '@/lib/logger'
import { ChevronDown } from 'lucide-react'

interface FormSectionProps {
  group: string
  values: Record<string, string>
  onChange: (name: string, value: string) => void
}

export default function FormSection({ group, values, onChange }: FormSectionProps) {
  const [open, setOpen] = useState(true)
  const { t } = useLanguage()
  const fields = FIELDS.filter((f) => f.group === group)
  const title = t.groups[group] ?? group

  logger.log(`Rendering FormSection: ${group} (${fields.length} fields, open=${open})`)

  return (
    <Card>
      <CardHeader
        className="flex flex-row items-center justify-between cursor-pointer select-none"
        onClick={() => setOpen((prev) => !prev)}
      >
        <CardTitle className="text-lg">{title}</CardTitle>
        <ChevronDown
          className={`h-5 w-5 text-muted-foreground transition-transform duration-200 ${
            open ? '' : '-rotate-90'
          }`}
        />
      </CardHeader>
      {open && (
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {fields.map((field) => (
              <FormField
                key={field.name}
                field={field}
                value={values[field.name]}
                onChange={onChange}
              />
            ))}
          </div>
        </CardContent>
      )}
    </Card>
  )
}
