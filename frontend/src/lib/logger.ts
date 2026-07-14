const isDev = import.meta.env.VITE_ENV !== 'prod'

const noop = (..._args: unknown[]) => {}

export const logger = isDev
  ? {
      log: console.log.bind(console, '[LOG]'),
      info: console.info.bind(console, '[INFO]'),
      warn: console.warn.bind(console, '[WARN]'),
      error: console.error.bind(console, '[ERROR]'),
    }
  : {
      log: noop,
      info: noop,
      warn: noop,
      error: noop,
    }
