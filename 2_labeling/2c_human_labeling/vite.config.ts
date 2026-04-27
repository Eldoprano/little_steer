import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

const DATASET_PATH = path.resolve(__dirname, '../../data/dataset.jsonl')
const WORK_ORDER_PATH = path.resolve(__dirname, '../2b_sentence/work_order_iaa.json')

interface EntryMeta {
  id: string
  uid: string
  model: string
  dataset: string
  judges: string[]
  files: string[]
  source_file: string
  has_reasoning: boolean
}

interface MetaIndex {
  entries: EntryMeta[]
  built_at: string
}

let metaCache: MetaIndex | null = null

function buildMeta(): MetaIndex {
  const entries = new Map<string, EntryMeta>()

  if (fs.existsSync(DATASET_PATH)) {
    const content = fs.readFileSync(DATASET_PATH, 'utf-8')
    for (const line of content.split('\n')) {
      if (!line.trim()) continue
      try {
        const e = JSON.parse(line)
        if (!e.id) continue
        
        const m = e.metadata || {}
        const hasReasoning =
          Boolean(m.has_reasoning) ||
          (Array.isArray(e.messages) && e.messages.some((msg: { role: string }) => msg.role === 'reasoning'))
        
        const judges: string[] = []
        if (Array.isArray(e.label_runs)) {
          for (const run of e.label_runs) {
            if (run.judge_name && !judges.includes(run.judge_name)) {
              judges.push(run.judge_name)
            }
          }
        }
        // Also check legacy judge field
        if (e.judge && !judges.includes(e.judge)) judges.push(e.judge)
        if (m.judge_name && !judges.includes(m.judge_name)) judges.push(m.judge_name)

        const uid = `dataset.jsonl:${e.id}`
        entries.set(uid, {
          id: e.id,
          uid,
          model: e.model || m.model_name || 'unknown',
          dataset: m.dataset_name || 'unknown',
          judges,
          files: ['dataset.jsonl'],
          source_file: 'dataset.jsonl',
          has_reasoning: hasReasoning,
        })
      } catch {}
    }
  }

  return {
    entries: Array.from(entries.values()),
    built_at: new Date().toISOString(),
  }
}

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-data',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const url = req.url ?? ''

          // GET /api/meta → cross-referenced entry index
          if (url === '/api/meta' || url.startsWith('/api/meta?')) {
            try {
              if (!metaCache) metaCache = buildMeta()
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify(metaCache))
            } catch (err) {
              res.statusCode = 500
              res.end(JSON.stringify({ error: String(err) }))
            }
            return
          }

          // GET /api/humans → list of human labeler tags
          if (url === '/api/humans') {
            try {
              if (!metaCache) metaCache = buildMeta()
              const humans = new Set<string>()
              for (const e of metaCache.entries) {
                for (const j of e.judges) {
                  if (j.startsWith('human_')) humans.add(j)
                }
              }
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify([...humans].sort()))
            } catch (err) {
              res.statusCode = 500
              res.end(JSON.stringify({ error: String(err) }))
            }
            return
          }

          // POST /api/save → save human label run
          if (url === '/api/save' && req.method === 'POST') {
            let body = ''
            req.on('data', chunk => { body += chunk })
            req.on('end', () => {
              try {
                const labelRunEntry = JSON.parse(body)
                let id = labelRunEntry.id
                if (!id) throw new Error('Missing ID')

                // Strip the "dataset.jsonl:" prefix if it exists to match the original dataset ID
                if (id.startsWith('dataset.jsonl:')) {
                  id = id.substring('dataset.jsonl:'.length)
                }

                // 1. Read entire dataset
                const content = fs.readFileSync(DATASET_PATH, 'utf-8')
                const lines = content.split('\n')
                let found = false
                
                const newLines = lines.map(line => {
                  if (!line.trim()) return line
                  try {
                    const e = JSON.parse(line)
                    if (e.id === id) {
                      found = true
                      // Append to label_runs
                      if (!e.label_runs) e.label_runs = []
                      
                      // Convert the flat human label entry back to a LabelRun structure
                      const newRun = {
                        judge_name: labelRunEntry.judge,
                        judge_model_id: 'human',
                        taxonomy_version: labelRunEntry.metadata.taxonomy_version || 'v6',
                        labeled_at: labelRunEntry.metadata.labeled_at,
                        generation_hash: e.generation_hash || '',
                        reasoning_truncated: false,
                        assessment: labelRunEntry.metadata.assessment,
                        spans: labelRunEntry.annotations,
                        status: labelRunEntry.metadata.status || 'completed'
                      }
                      
                      // Remove existing run from same human if any (overwrite)
                      e.label_runs = e.label_runs.filter((r: any) => r.judge_name !== newRun.judge_name)
                      e.label_runs.push(newRun)

                      // If marked as broken, update metadata so fix_quality.py can see it
                      if (newRun.status === 'error') {
                        e.metadata.approved = false
                        if (!e.metadata.quality) {
                          e.metadata.quality = {
                            issues: [],
                            approved: false,
                            checked_at: new Date().toISOString()
                          }
                        }
                        e.metadata.quality.approved = false
                        if (!e.metadata.quality.issues.includes('human_broken')) {
                          e.metadata.quality.issues.push('human_broken')
                        }
                      }

                      return JSON.stringify(e)
                    }
                  } catch {}
                  return line
                })

                if (!found) throw new Error(`Entry ${id} not found in dataset`)

                fs.writeFileSync(DATASET_PATH, newLines.join('\n'), 'utf-8')
                metaCache = null // Invalidate cache
                
                res.setHeader('Content-Type', 'application/json')
                res.end(JSON.stringify({ ok: true }))
              } catch (err) {
                res.statusCode = 500
                res.end(JSON.stringify({ error: String(err) }))
              }
            })
            return
          }

          // GET /api/data/dataset.jsonl → serve dataset.jsonl
          if (url === '/api/data/dataset.jsonl' || url === '/api/gen-data/dataset.jsonl') {
            if (!fs.existsSync(DATASET_PATH)) {
              res.statusCode = 404
              res.end(JSON.stringify({ error: 'dataset.jsonl not found' }))
              return
            }
            res.setHeader('Content-Type', 'application/x-ndjson')
            fs.createReadStream(DATASET_PATH).pipe(res)
            return
          }

          // GET /api/work-order → serve flat_order from work_order_iaa.json
          if (url === '/api/work-order') {
            if (!fs.existsSync(WORK_ORDER_PATH)) {
              res.statusCode = 404
              res.end(JSON.stringify({ error: 'work_order_iaa.json not found' }))
              return
            }
            res.setHeader('Content-Type', 'application/json')
            fs.createReadStream(WORK_ORDER_PATH).pipe(res)
            return
          }

          // GET /api/entry/:id → fetch a single entry from dataset.jsonl
          if (url.startsWith('/api/entry/')) {
            const id = url.replace('/api/entry/', '')
            if (!id) {
              res.statusCode = 400
              res.end(JSON.stringify({ error: 'Missing ID' }))
              return
            }

            try {
              const content = fs.readFileSync(DATASET_PATH, 'utf-8')
              const lines = content.split('\n')
              const line = lines.find(l => {
                if (!l.trim()) return false
                try {
                  const e = JSON.parse(l)
                  return e.id === id
                } catch { return false }
              })

              if (!line) {
                res.statusCode = 404
                res.end(JSON.stringify({ error: 'Entry not found' }))
                return
              }

              res.setHeader('Content-Type', 'application/json')
              res.end(line)
            } catch (err) {
              res.statusCode = 500
              res.end(JSON.stringify({ error: String(err) }))
            }
            return
          }

          next()
        })
      },
    },
  ],
  server: {
    port: 5173,
    strictPort: true,
    host: '0.0.0.0',
    allowedHosts: true,
    cors: true,
    fs: {
      allow: [path.resolve(__dirname, '../../')],
    },
    hmr: false,
    watch: {
      ignored: [
        `${path.resolve(__dirname, '../../data')}/**`,
        `${path.resolve(__dirname, '../2b_sentence')}/**`,
      ],
    },
  },
})
