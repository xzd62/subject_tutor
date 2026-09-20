<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, Promotion, RefreshLeft } from '@element-plus/icons-vue'
import { ChatSocket } from '../api/ws'
import { http } from '../api/http'
import type { ChatEvent, RetrievalData, SessionHistory, SessionItem } from '../api/types'
import MarkdownText from '../components/MarkdownText.vue'

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
  retrieval?: RetrievalData | null
  status?: string
  error?: string
  cache?: { hit_type: string; score: number; cache_id: string }
}

const STORAGE_KEY = 'tutor-thread-id'
const messages = ref<ChatMessage[]>([])
const input = ref('')
const streaming = ref(false)
const connected = ref(false)
const scroller = ref<HTMLElement | null>(null)
const sessions = ref<SessionItem[]>([])
const threadId = ref('')
const CACHE_KEY = 'tutor-cache-enabled'
const cacheEnabled = ref(localStorage.getItem(CACHE_KEY) !== '0')
const bypassCacheOnce = ref(false)

function persistCacheSetting(): void {
  localStorage.setItem(CACHE_KEY, cacheEnabled.value ? '1' : '0')
}

const socket = new ChatSocket()

const sessionOptions = computed(() => {
  const list = sessions.value.map((session) => ({
    ...session,
    title: session.title || session.thread_id.slice(0, 8),
  }))
  if (threadId.value && !list.some((session) => session.thread_id === threadId.value)) {
    list.unshift({
      thread_id: threadId.value,
      title: '新对话',
      created_at: '',
      updated_at: '',
    })
  }
  return list
})

function persistThread(): void {
  localStorage.setItem(STORAGE_KEY, threadId.value)
}

function lastAssistant(): ChatMessage | undefined {
  const last = messages.value[messages.value.length - 1]
  return last?.role === 'assistant' ? last : undefined
}

async function scrollToBottom(): Promise<void> {
  await nextTick()
  if (scroller.value) scroller.value.scrollTop = scroller.value.scrollHeight
}

async function loadSessions(): Promise<void> {
  const data = await http.get<{ items: SessionItem[] }>('/api/sessions')
  sessions.value = data.items
}

async function loadHistory(id: string): Promise<void> {
  const data = await http.get<SessionHistory>(`/api/sessions/${id}/messages`)
  messages.value = data.messages.map((message) => ({
    role: message.role,
    content: message.content,
  }))
  void scrollToBottom()
}

async function bootstrap(): Promise<void> {
  await loadSessions()
  const stored = localStorage.getItem(STORAGE_KEY)
  const exists = (id: string) => sessions.value.some((session) => session.thread_id === id)
  if (stored && exists(stored)) {
    threadId.value = stored
  } else if (sessions.value.length) {
    threadId.value = sessions.value[0].thread_id
  } else {
    threadId.value = crypto.randomUUID()
  }
  persistThread()
  if (exists(threadId.value)) {
    await loadHistory(threadId.value)
  }
}

function handleEvent(event: ChatEvent): void {
  if (event.type === 'status') {
    const message = lastAssistant()
    if (!message) return
    if (event.stage === 'retrieving') {
      message.status = '正在检索教材…'
    } else if (event.stage === 'thinking') {
      message.status = '正在思考…'
    } else if (event.stage === 'subagent') {
      message.status = `正在咨询${event.data?.subject ?? ''}老师…`
    } else {
      const toolName = event.data?.tool ?? ''
      const memoryName = event.data?.args?.name ?? ''
      message.status =
        toolName === 'memory_read' ? `正在查看记忆：${memoryName}` : `正在记录记忆：${memoryName}`
    }
    return
  }
  if (event.type === 'retrieval') {
    const message = lastAssistant()
    if (message) {
      message.retrieval = event.data
      message.status = ''
    }
    return
  }
  if (event.type === 'cache') {
    const message = lastAssistant()
    if (message) {
      message.status = ''
      if (event.data.hit) {
        message.cache = {
          hit_type: event.data.hit_type ?? 'exact',
          score: event.data.score ?? 1,
          cache_id: event.data.cache_id ?? '',
        }
        if (event.data.sources && event.data.sources.length) {
          message.retrieval = {
            query: '',
            top_k: event.data.sources.length,
            degraded: null,
            elapsed_ms: {},
            hits: event.data.sources,
          }
        }
      }
    }
    return
  }
  if (event.type === 'token') {
    const message = lastAssistant()
    if (message) {
      message.status = ''
      message.content += event.text
    }
    void scrollToBottom()
    return
  }
  if (event.type === 'done') {
    streaming.value = false
    const message = lastAssistant()
    if (message) message.status = ''
    void loadSessions()
    return
  }
  if (event.type === 'error') {
    streaming.value = false
    const message = lastAssistant()
    if (message) {
      message.status = ''
      message.error = event.message
    }
  }
}

function send(): void {
  const text = input.value.trim()
  if (!text || streaming.value) return
  if (!socket.isOpen) {
    ElMessage.warning('连接尚未就绪，请稍候重试')
    return
  }
  messages.value.push({ role: 'user', content: text })
  messages.value.push({ role: 'assistant', content: '', status: '正在思考…' })
  input.value = ''
  streaming.value = true
  try {
    socket.send({
      message: text,
      thread_id: threadId.value,
      use_cache: cacheEnabled.value && !bypassCacheOnce.value,
    })
  } catch (error) {
    streaming.value = false
    ElMessage.error((error as Error).message)
  } finally {
    bypassCacheOnce.value = false
  }
  void scrollToBottom()
}

async function reanswer(index: number): Promise<void> {
  const assistant = messages.value[index]
  const userMessage = messages.value[index - 1]
  if (!assistant || assistant.role !== 'assistant' || !userMessage || userMessage.role !== 'user') {
    return
  }
  const cacheId = assistant.cache?.cache_id
  if (cacheId) {
    try {
      await http.del(`/api/cache/${encodeURIComponent(cacheId)}`)
    } catch {
      // 缓存可能已过期，继续重新生成
    }
  }
  messages.value.splice(index - 1, 2)
  input.value = userMessage.content
  bypassCacheOnce.value = true
  send()
}

function stop(): void {
  socket.close()
  streaming.value = false
  const message = lastAssistant()
  if (message) message.status = ''
  socket.connect()
}

async function selectSession(id: string | number): Promise<void> {
  const target = String(id)
  if (target === threadId.value) return
  if (streaming.value) {
    ElMessage.warning('正在生成回复，请先等待或点击停止')
    return
  }
  threadId.value = target
  persistThread()
  await loadHistory(target)
}

async function removeSession(id: string): Promise<void> {
  if (streaming.value && id === threadId.value) {
    ElMessage.warning('正在生成回复，暂不能删除当前会话')
    return
  }
  try {
    await ElMessageBox.confirm('确定删除这个会话？聊天记录将一并清除。', '删除会话', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
  } catch {
    return
  }
  await http.del(`/api/sessions/${id}`)
  ElMessage.success('会话已删除')
  await loadSessions()
  if (id === threadId.value) {
    newConversation()
  }
}

function newConversation(): void {
  threadId.value = crypto.randomUUID()
  persistThread()
  messages.value = []
  streaming.value = false
}

onMounted(async () => {
  socket.on(handleEvent)
  socket.onOpen(() => {
    connected.value = true
  })
  socket.onClose(() => {
    connected.value = false
  })
  socket.connect()
  try {
    await bootstrap()
  } catch (error) {
    ElMessage.error(`会话加载失败：${(error as Error).message}`)
  }
})
onBeforeUnmount(() => socket.close())
</script>

<template>
  <div class="chat-view">
    <div class="chat-toolbar">
      <div class="chat-title">AI 对话</div>
      <div class="chat-actions">
        <el-switch
          v-model="cacheEnabled"
          size="small"
          active-text="语义缓存"
          @change="persistCacheSetting"
        />
        <el-tag :type="connected ? 'success' : 'info'" size="small">
          {{ connected ? '已连接' : '连接中' }}
        </el-tag>
        <el-select
          :model-value="threadId"
          size="small"
          class="session-select"
          placeholder="选择会话"
          @change="selectSession"
        >
          <el-option
            v-for="session in sessionOptions"
            :key="session.thread_id"
            :label="session.title"
            :value="session.thread_id"
          >
            <div class="session-option">
              <span class="session-title">{{ session.title }}</span>
              <el-icon
                v-if="sessions.some((item) => item.thread_id === session.thread_id)"
                class="session-delete"
                @click.stop="removeSession(session.thread_id)"
              >
                <Delete />
              </el-icon>
            </div>
          </el-option>
        </el-select>
        <el-button size="small" :icon="RefreshLeft" @click="newConversation">新对话</el-button>
      </div>
    </div>

    <div ref="scroller" class="chat-messages">
      <el-empty
        v-if="!messages.length"
        description="向家教老师提问吧，比如：有理数的加法法则是什么？"
      />
      <div
        v-for="(message, index) in messages"
        :key="index"
        :class="['bubble-row', message.role]"
      >
        <div class="bubble">
          <template v-if="message.role === 'assistant'">
            <div v-if="message.cache" class="cache-line">
              <el-tag type="warning" size="small">
                ⚡ 来自缓存（相似度 {{ Math.round((message.cache.score ?? 1) * 100) }}%）
              </el-tag>
              <el-button link type="primary" size="small" @click="reanswer(index)">
                重新回答
              </el-button>
            </div>
            <div v-if="message.status" class="status-line">{{ message.status }}</div>
            <MarkdownText v-if="message.content" :content="message.content" />
            <el-alert
              v-if="message.error"
              :title="message.error"
              type="error"
              :closable="false"
              show-icon
            />
            <el-collapse
              v-if="message.retrieval && message.retrieval.hits.length"
              class="retrieval-panel"
            >
              <el-collapse-item
                :title="`教材依据（${message.retrieval.hits.length} 条，检索 ${(message.retrieval.elapsed_ms.total ?? 0).toFixed(0)} ms）`"
              >
                <div
                  v-for="(hit, hitIndex) in message.retrieval.hits"
                  :key="hit.chunk_id"
                  class="retrieval-hit"
                >
                  <div class="retrieval-head">
                    <span class="retrieval-index">[{{ hitIndex + 1 }}]</span>
                    <span class="retrieval-path">
                      {{ hit.subject }} · {{ hit.heading_path || hit.source_file }}
                    </span>
                  </div>
                  <div class="retrieval-preview">{{ hit.preview }}</div>
                  <div class="retrieval-score">
                    RRF {{ hit.rrf_score.toFixed(4) }} · 向量#{{ hit.vector_rank ?? '-' }} ·
                    BM25#{{ hit.bm25_rank ?? '-' }}
                  </div>
                </div>
              </el-collapse-item>
            </el-collapse>
          </template>
          <template v-else>{{ message.content }}</template>
        </div>
      </div>
    </div>

    <div class="chat-input">
      <el-input
        v-model="input"
        type="textarea"
        :rows="3"
        resize="none"
        placeholder="输入你的问题，Enter 发送，Shift+Enter 换行"
        @keydown.enter.exact.prevent="send"
      />
      <div class="chat-input-actions">
        <el-button v-if="streaming" type="danger" plain @click="stop">停止</el-button>
        <el-button v-else type="primary" :icon="Promotion" @click="send">发送</el-button>
      </div>
    </div>
  </div>
</template>
