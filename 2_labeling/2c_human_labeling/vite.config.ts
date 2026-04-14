import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'

const GEN_DIR = path.resolve(__dirname, '../../data/1_generated')
const LABELED_DIR = path.resolve(__dirname, '../../data/2b_labeled')
const WORK_ORDER_PATH = path.resolve(__dirname, '../2b_sentence/work_order.json')

interface EntryMeta {
  id: string         // original prompt ID (shared across models)
  uid: string        // globally unique: source_file + ':' + id
  model: string
  dataset: string
  judges: string[]
  files: string[]        // filenames in LABELED_DIR that contain this entry
  source_file: string    // filename in GEN_DIR where this entry originated
  has_reasoning: boolean
}

interface MetaIndex {
  entries: EntryMeta[]
  built_at: string
}

let metaCache: MetaIndex | null = null

function buildMeta(): MetaIndex {
  // Key: uid = source_file + ':' + id  →  each (model, prompt) pair is distinct
  const entries = new Map<string, EntryMeta>()

  // Scan 1_generated — every (file, entry) combo gets its own record
  const sourceFiles: string[] = []
  if (fs.existsSync(GEN_DIR)) {
    for (const file of fs.readdirSync(GEN_DIR).sort()) {
      if (!file.endsWith('.jsonl')) continue
      sourceFiles.push(file)
      const content = fs.readFileSync(path.join(GEN_DIR, file), 'utf-8')
      for (const line of content.split('\n')) {
        if (!line.trim()) continue
        try {
          const e = JSON.parse(line)
          if (!e.id) continue
          const uid = `${file}:${e.id}`
          if (entries.has(uid)) continue  // dedup within same file only
          const m = e.metadata || {}
          const hasReasoning =
            Boolean(m.has_reasoning) ||
            (Array.isArray(e.messages) && e.messages.some((msg: { role: string }) => msg.role === 'reasoning'))
          entries.set(uid, {
            id: e.id,
            uid,
            model: m.model_name || 'unknown',
            dataset: m.dataset_name || 'unknown',
            judges: [],
            files: [],
            source_file: file,
            has_reasoning: hasReasoning,
          })
        } catch {}
      }
    }
  }

  // Scan 2b_labeled — match each labeled file back to its source file in GEN_DIR.
  // Labeled filenames follow the pattern: {source_without_ext}_{judge}.jsonl
  // We find the source by looking for the longest known source prefix.
  if (fs.existsSync(LABELED_DIR)) {
    for (const file of fs.readdirSync(LABELED_DIR).sort()) {
      if (!file.endsWith('.jsonl')) continue
      const fileBase = file.slice(0, -6)

      let sourceFile = ''
      for (const sf of sourceFiles) {
        const sfBase = sf.slice(0, -6)
        if ((fileBase === sfBase || fileBase.startsWith(sfBase + '_')) && sfBase.length > sourceFile.length) {
          sourceFile = sf
        }
      }
      if (!sourceFile) continue

      const content = fs.readFileSync(path.join(LABELED_DIR, file), 'utf-8')
      for (const line of content.split('\n')) {
        if (!line.trim()) continue
        try {
          const e = JSON.parse(line)
          if (!e.id) continue
          const uid = `${sourceFile}:${e.id}`
          const meta = entries.get(uid)
          if (!meta) continue
          const judge = e.metadata?.judge_name || e.judge || ''
          if (judge && !meta.judges.includes(judge)) meta.judges.push(judge)
          if (!meta.files.includes(file)) meta.files.push(file)
        } catch {}
      }
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

          // GET /api/meta/refresh → force rebuild
          if (url === '/api/meta/refresh') {
            try {
              metaCache = buildMeta()
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify({
                ok: true,
                built_at: metaCache.built_at,
                count: metaCache.entries.length,
                debug: {
                  gen_dir: GEN_DIR,
                  labeled_dir: LABELED_DIR,
                  gen_dir_exists: fs.existsSync(GEN_DIR),
                  labeled_dir_exists: fs.existsSync(LABELED_DIR),
                  gen_files: fs.existsSync(GEN_DIR) ? fs.readdirSync(GEN_DIR).filter(f => f.endsWith('.jsonl')).length : 0,
                },
              }))
            } catch (err) {
              res.statusCode = 500
              res.end(JSON.stringify({ error: String(err) }))
            }
            return
          }

          // GET /api/gen-data/<filename> → serve raw entry from GEN_DIR
          if (url.startsWith('/api/gen-data/')) {
            const filename = decodeURIComponent(url.slice('/api/gen-data/'.length))
            if (!filename || !filename.endsWith('.jsonl') || filename.includes('..')) {
              res.statusCode = 400
              res.end(JSON.stringify({ error: 'Invalid filename' }))
              return
            }
            const filePath = path.join(GEN_DIR, filename)
            if (!fs.existsSync(filePath)) {
              res.statusCode = 404
              res.end(JSON.stringify({ error: 'File not found' }))
              return
            }
            res.setHeader('Content-Type', 'application/x-ndjson')
            fs.createReadStream(filePath).pipe(res)
            return
          }

          // GET /api/data/<filename> → serve from LABELED_DIR
          if (url.startsWith('/api/data/')) {
            const filename = decodeURIComponent(url.slice('/api/data/'.length))
            if (!filename || !filename.endsWith('.jsonl') || filename.includes('..')) {
              res.statusCode = 400
              res.end(JSON.stringify({ error: 'Invalid filename' }))
              return
            }
            const filePath = path.join(LABELED_DIR, filename)
            if (!fs.existsSync(filePath)) {
              res.statusCode = 404
              res.end(JSON.stringify({ error: 'File not found' }))
              return
            }
            res.setHeader('Content-Type', 'application/x-ndjson')
            fs.createReadStream(filePath).pipe(res)
            return
          }

          // GET /api/work-order → serve flat_order from work_order.json
          if (url === '/api/work-order') {
            if (!fs.existsSync(WORK_ORDER_PATH)) {
              res.statusCode = 404
              res.end(JSON.stringify({ error: 'work_order.json not found' }))
              return
            }
            res.setHeader('Content-Type', 'application/json')
            fs.createReadStream(WORK_ORDER_PATH).pipe(res)
            return
          }

          // GET /api/files → backward-compat file list
          if (url === '/api/files') {
            try {
              const files = fs.existsSync(LABELED_DIR)
                ? fs.readdirSync(LABELED_DIR).filter(f => f.endsWith('.jsonl')).sort()
                : []
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify(files))
            } catch {
              res.statusCode = 500
              res.end(JSON.stringify({ error: 'Could not read data directory' }))
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
    // Disable HMR WebSocket to prevent page reloads through Cloudflare tunnel
    hmr: false,
    watch: {
      // Exclude large data dirs from file watching to avoid spurious restarts
      ignored: [
        `${path.resolve(__dirname, '../../data')}/**`,
        `${path.resolve(__dirname, '../../.stfolder')}/**`,
        `${path.resolve(__dirname, '../2b_sentence')}/**`,
      ],
    },
  },
})
