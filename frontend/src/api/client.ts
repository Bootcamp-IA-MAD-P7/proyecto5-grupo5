import axios from 'axios'
import { logger } from '@/lib/logger'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5000',
})

api.interceptors.request.use((config) => {
  logger.log(`API request: ${config.method?.toUpperCase()} ${config.url}`, config.data)
  return config
})

api.interceptors.response.use(
  (response) => {
    logger.log(`API response: ${response.status} from ${response.config.url}`, response.data)
    return response
  },
  (error) => {
    logger.error(`API error: ${error.response?.status} ${error.config?.url}`, error.message)
    return Promise.reject(error)
  },
)

export default api
