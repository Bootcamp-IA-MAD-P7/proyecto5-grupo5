# Frontend — Churn Predictor

Dashboard para predecir abandono de clientes. Envía las 19 variables del dataset a una API y visualiza el resultado con nivel de riesgo y probabilidad.

## Tecnologías

- **React 19** + **TypeScript**
- **Vite** (build tool)
- **Tailwind CSS v4** + **shadcn/ui** (componentes)
- **React Router** (navegación)
- **Axios** (HTTP client)
- **Lucide React** (iconos)

## Instalación

```bash
cd frontend
npm install
```

## Entorno

Copiar `.env.example` a `.env` y configurar:

| Variable | Descripción | Default |
|----------|-------------|---------|
| `VITE_API_URL` | URL del backend | `http://localhost:5000` |
| `VITE_ENV` | `dev` o `prod` | `dev` |
| `VITE_MOCK` | Usar mock en vez de API real | `false` |

## Scripts

```bash
npm run dev      # servidor de desarrollo
npm run build    # build de producción
npm run preview  # previsualizar build
```
