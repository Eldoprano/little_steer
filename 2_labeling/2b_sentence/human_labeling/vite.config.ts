import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

// Labeled data directory (relative to this config file)
const DATA_DIR = path.resolve(__dirname, '../../../data/2b_labeled')

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-labeled-data',
      configureServer(server) {
        // GET /api/files → list .jsonl files in the data directory
        server.middlewares.use('/api/files', (_req, res) => {
          try {
            const files = fs.readdirSync(DATA_DIR)
              .filter(f => f.endsWith('.jsonl'))
              .sort()
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify(files))
          } catch {
            res.statusCode = 500
            res.end(JSON.stringify({ error: 'Could not read data directory' }))
          }
        })

        // GET /api/data/<filename> → serve a specific .jsonl file
        server.middlewares.use('/api/data/', (req, res) => {
          const filename = decodeURIComponent(req.url?.replace(/^\//, '') ?? '')
          if (!filename || !filename.endsWith('.jsonl') || filename.includes('..')) {
            res.statusCode = 400
            res.end(JSON.stringify({ error: 'Invalid filename' }))
            return
          }
          const filePath = path.join(DATA_DIR, filename)
          if (!fs.existsSync(filePath)) {
            res.statusCode = 404
            res.end(JSON.stringify({ error: 'File not found' }))
            return
          }
          res.setHeader('Content-Type', 'application/x-ndjson')
          fs.createReadStream(filePath).pipe(res)
        })
      },
    },
  ],
  server: {
    port: 5173,
    host: true,
  },
})
