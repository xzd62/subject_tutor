export interface RetrievalHit {
  chunk_id: string
  subject: string
  grade: string
  source_file: string
  heading_path: string
  rrf_score: number
  vector_rank: number | null
  bm25_rank: number | null
  preview: string
}

export interface RetrievalData {
  query: string
  top_k: number
  degraded: string | null
  elapsed_ms: Record<string, number>
  hits: RetrievalHit[]
}

export type ChatEvent =
  | { type: 'status'; stage: 'retrieving' }
  | { type: 'status'; stage: 'thinking' }
  | {
      type: 'status'
      stage: 'memory'
      data: { tool: string; args: Record<string, string> }
    }
  | { type: 'status'; stage: 'subagent'; data: { subject: string } }
  | {
      type: 'cache'
      data: {
        hit: boolean
        hit_type?: string
        score?: number
        cache_id?: string
        sources?: RetrievalHit[]
      }
    }
  | { type: 'retrieval'; data: RetrievalData }
  | { type: 'token'; text: string }
  | { type: 'done' }
  | { type: 'error'; message: string }
  | { type: 'reset_ok' }

export interface CacheStats {
  enabled: boolean
  connected: boolean
  namespace: string
  threshold: number
  protect_numeric: boolean
  entries: number
  hit_rate: number | null
  counters: Record<string, number>
}

export interface MemoryItem {
  name: string
  type: string
  description: string
  updated_at: string
  damaged: boolean
}

export interface MemoryDetail extends MemoryItem {
  body: string
  raw?: string
}

export interface ModelSettings {
  model: string
  temperature: number
  max_tokens: number
  timeout_s: number
  max_retries: number
  api_key: string
  model_ready: boolean
  model_error: string | null
}

export interface ModelPreset {
  provider: string
  label: string
  models: string[]
  key_env: string
  installed: boolean
  note: string
}

export interface TextbookItem {
  doc_id: string
  source_file: string
  subject: string
  grade: string
  textbook_version: string
  chunk_count: number
  actual_chunks: number
  ingested_at: string
  pending_rebuild: boolean
}

export interface ScheduleState {
  time: string
  trigger: string
  rebuilt: string[]
  rebuilt_count: number
  skipped_count: number
  failed_count: number
  elapsed_s: number
  corpus_size?: number
  collection_chunks?: number
}

export interface SessionItem {
  thread_id: string
  title: string
  created_at: string
  updated_at: string
}

export interface SessionHistory {
  thread_id: string
  title: string
  messages: { role: 'user' | 'assistant'; content: string }[]
}

export interface TextbookChunk {
  chunk_id: string
  chunk_index: number
  heading_path: string
  char_count: number
  text: string
}

export interface TextbookDetail {
  doc_id: string
  source_file: string
  source_format: string
  subject: string
  grade: string
  textbook_version: string
  pending_rebuild: boolean
  chunks: TextbookChunk[]
}

export interface UploadResult {
  ok: boolean
  doc_id: string
  source_file: string
  chunks: number
  collection_chunks: number
  ingested: string[]
  skipped: string[]
  failed: { file: string; stage: string; error: string }[]
  elapsed_s: number
}
