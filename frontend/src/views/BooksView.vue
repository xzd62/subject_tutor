<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, Download, Edit, Refresh, UploadFilled, View } from '@element-plus/icons-vue'
import { http } from '../api/http'
import type {
  ScheduleState,
  TextbookDetail,
  TextbookItem,
  UploadResult,
} from '../api/types'

const SUBJECTS = ['语文', '数学', '英语', '物理', '化学', '生物', '政治', '历史', '地理']
const GRADES = ['七上', '七下', '八上', '八下', '九上', '九下']
const VERSIONS = ['人教版', '部编版', '苏教版', '北师大版', '沪教版', '外研版', '译林版', '湘教版']
const EDITABLE_FORMATS = ['md', 'markdown', 'txt']

const form = reactive({ subject: '数学', grade: '七上', version: '人教版' })
const file = ref<File | null>(null)
const uploading = ref(false)
const result = ref<UploadResult | null>(null)
const textbooks = ref<TextbookItem[]>([])
const collectionChunks = ref(0)
const uploadRef = ref()

const drawerVisible = ref(false)
const detailLoading = ref(false)
const detail = ref<TextbookDetail | null>(null)
const editing = ref(false)
const editContent = ref('')
const editLoading = ref(false)

const lastCheck = ref<ScheduleState | null>(null)
const checking = ref(false)
const rebuildingId = ref('')

function onFileChange(uploadFile: { raw?: File }): void {
  file.value = uploadFile.raw ?? null
}

async function loadTextbooks(): Promise<void> {
  const data = await http.get<{ items: TextbookItem[]; collection_chunks: number }>(
    '/api/textbooks',
  )
  textbooks.value = data.items
  collectionChunks.value = data.collection_chunks
}

async function loadSchedule(): Promise<void> {
  const data = await http.get<{ last_run: ScheduleState | null }>('/api/schedule')
  lastCheck.value = data.last_run && data.last_run.time ? data.last_run : null
}

async function submit(): Promise<void> {
  if (!file.value) {
    ElMessage.warning('请先选择教材文件')
    return
  }
  uploading.value = true
  result.value = null
  try {
    const payload = new FormData()
    payload.append('file', file.value)
    payload.append('subject', form.subject)
    payload.append('grade', form.grade)
    payload.append('version', form.version)
    const response = await fetch('/api/upload/textbook', { method: 'POST', body: payload })
    const data = await response.json()
    if (!response.ok) {
      throw new Error(typeof data.detail === 'string' ? data.detail : '上传失败')
    }
    result.value = data as UploadResult
    ElMessage.success(`入库成功：${data.chunks} 个知识块，当次会话即可检索`)
    file.value = null
    uploadRef.value?.clearFiles()
    await loadTextbooks()
  } catch (error) {
    ElMessage.error((error as Error).message)
  } finally {
    uploading.value = false
  }
}

async function viewBook(row: TextbookItem): Promise<void> {
  drawerVisible.value = true
  detailLoading.value = true
  detail.value = null
  editing.value = false
  try {
    detail.value = await http.get<TextbookDetail>(`/api/textbooks/${row.doc_id}/chunks`)
  } catch (error) {
    ElMessage.error((error as Error).message)
    drawerVisible.value = false
  } finally {
    detailLoading.value = false
  }
}

function downloadBook(row: { doc_id: string }): void {
  window.open(`/api/textbooks/${row.doc_id}/file`, '_blank')
}

async function startEdit(): Promise<void> {
  if (!detail.value) return
  editLoading.value = true
  try {
    const data = await http.get<{ content: string }>(
      `/api/textbooks/${detail.value.doc_id}/content`,
    )
    editContent.value = data.content
    editing.value = true
  } catch (error) {
    ElMessage.error((error as Error).message)
  } finally {
    editLoading.value = false
  }
}

async function saveEdit(): Promise<void> {
  if (!detail.value) return
  editLoading.value = true
  try {
    await http.put(`/api/textbooks/${detail.value.doc_id}/content`, {
      content: editContent.value,
    })
    ElMessage.success('内容已保存，请点击「重建向量」使其生效')
    editing.value = false
    await loadTextbooks()
  } catch (error) {
    ElMessage.error((error as Error).message)
  } finally {
    editLoading.value = false
  }
}

async function rebuildBook(row: TextbookItem): Promise<void> {
  try {
    await ElMessageBox.confirm(
      `确定重新切块并重建《${row.source_file}》的向量索引？重建期间该教材短暂不可检索。`,
      '重建向量',
      { type: 'warning', confirmButtonText: '重建', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  rebuildingId.value = row.doc_id
  try {
    const data = await http.post<{ removed_chunks: number; chunk_count: number }>(
      `/api/textbooks/${row.doc_id}/rebuild`,
    )
    ElMessage.success(`重建完成：移除旧块 ${data.removed_chunks} 个，现有 ${data.chunk_count} 个`)
    await loadTextbooks()
  } catch (error) {
    ElMessage.error((error as Error).message)
  } finally {
    rebuildingId.value = ''
  }
}

async function recheck(): Promise<void> {
  checking.value = true
  try {
    const data = await http.post<ScheduleState>('/api/textbooks/recheck')
    if (data.rebuilt_count) {
      ElMessage.success(`检查完成：重建 ${data.rebuilt_count} 本（跳过 ${data.skipped_count} 本）`)
    } else {
      ElMessage.success(`检查完成：${data.skipped_count} 本教材均无变化`)
    }
    await Promise.all([loadTextbooks(), loadSchedule()])
  } catch (error) {
    ElMessage.error((error as Error).message)
  } finally {
    checking.value = false
  }
}

async function removeBook(row: TextbookItem): Promise<void> {
  try {
    await ElMessageBox.confirm(
      `确定删除《${row.source_file}》？将同时移除其 ${row.actual_chunks} 个知识块。`,
      '删除教材',
      { type: 'warning', confirmButtonText: '删除', cancelButtonText: '取消' },
    )
  } catch {
    return
  }
  try {
    const data = await http.del<{ removed_chunks: number }>(`/api/textbooks/${row.doc_id}`)
    ElMessage.success(`已删除（移除 ${data.removed_chunks} 个知识块）`)
    await loadTextbooks()
  } catch (error) {
    ElMessage.error((error as Error).message)
  }
}

onMounted(async () => {
  await Promise.all([loadTextbooks(), loadSchedule()])
})
</script>

<template>
  <div class="page">
    <h2>教材管理</h2>
    <el-card shadow="never">
      <el-form label-width="90px">
        <el-form-item label="学科">
          <el-select v-model="form.subject" style="width: 200px">
            <el-option v-for="item in SUBJECTS" :key="item" :label="item" :value="item" />
          </el-select>
        </el-form-item>
        <el-form-item label="年级">
          <el-select v-model="form.grade" style="width: 200px">
            <el-option v-for="item in GRADES" :key="item" :label="item" :value="item" />
          </el-select>
        </el-form-item>
        <el-form-item label="教材版本">
          <el-select
            v-model="form.version"
            filterable
            allow-create
            style="width: 200px"
            placeholder="可选择或输入版本"
          >
            <el-option v-for="item in VERSIONS" :key="item" :label="item" :value="item" />
          </el-select>
        </el-form-item>
        <el-form-item label="教材文件">
          <el-upload
            ref="uploadRef"
            :auto-upload="false"
            :limit="1"
            accept=".md,.markdown,.txt,.pdf"
            :on-change="onFileChange"
            drag
          >
            <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
            <div class="el-upload__text">拖拽文件到此处，或<em>点击选择</em></div>
            <template #tip>
              <div class="el-upload__tip">支持 md / txt / pdf，单个文件不超过 50MB</div>
            </template>
          </el-upload>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="uploading" @click="submit">
            {{ uploading ? '正在切块并构建索引…' : '上传并构建索引' }}
          </el-button>
        </el-form-item>
      </el-form>

      <el-alert v-if="result" type="success" :closable="false" show-icon>
        <template #title>
          入库「{{ result.source_file }}」：{{ result.chunks }} 块（检索库共
          {{ result.collection_chunks }} 块，耗时 {{ result.elapsed_s }}s）
        </template>
      </el-alert>
    </el-card>

    <el-card shadow="never" class="table-card">
      <template #header>
        <div class="card-header">
          <span>已入库教材</span>
          <div class="table-actions">
            <span class="muted">
              上次自动检查：
              {{ lastCheck ? `${lastCheck.time}（重建 ${lastCheck.rebuilt_count} 本）` : '尚未执行' }}
            </span>
            <el-button size="small" :icon="Refresh" :loading="checking" @click="recheck">
              检查全部更新
            </el-button>
          </div>
        </div>
      </template>
      <el-table :data="textbooks" size="small" empty-text="还没有上传教材">
        <el-table-column label="文件" min-width="220">
          <template #default="{ row }">
            <span>{{ row.source_file }}</span>
            <el-tag v-if="row.pending_rebuild" type="warning" size="small" class="pending-tag">
              待重建
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="subject" label="学科" width="70" />
        <el-table-column prop="grade" label="年级" width="70" />
        <el-table-column prop="textbook_version" label="版本" width="90" />
        <el-table-column prop="actual_chunks" label="知识块" width="80" />
        <el-table-column prop="ingested_at" label="入库时间" width="170" />
        <el-table-column label="操作" width="290" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" :icon="View" @click="viewBook(row)">查看</el-button>
            <el-button
              link
              type="warning"
              :icon="Refresh"
              :loading="rebuildingId === row.doc_id"
              @click="rebuildBook(row)"
            >
              重建向量
            </el-button>
            <el-button link type="primary" :icon="Download" @click="downloadBook(row)">
              下载
            </el-button>
            <el-button link type="danger" :icon="Delete" @click="removeBook(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-drawer
      v-model="drawerVisible"
      size="46%"
      :title="detail ? `教材详情 · ${detail.source_file}` : '教材详情'"
      @closed="editing = false"
    >
      <div v-loading="detailLoading || editLoading" class="detail-body">
        <template v-if="detail">
          <div class="detail-toolbar">
            <span class="muted">
              {{ detail.subject }} · {{ detail.grade }} · {{ detail.textbook_version }} ·
              {{ detail.chunks.length }} 个知识块
            </span>
            <div>
              <el-button
                v-if="EDITABLE_FORMATS.includes(detail.source_format)"
                size="small"
                :icon="Edit"
                :disabled="editing"
                @click="startEdit"
              >
                编辑内容
              </el-button>
              <el-button size="small" :icon="Download" @click="downloadBook(detail)">
                下载原文件
              </el-button>
            </div>
          </div>

          <template v-if="editing">
            <el-alert
              type="info"
              :closable="false"
              show-icon
              title="编辑后请点击列表中的「重建向量」使改动生效（夜间 00:00 也会自动兜底检查）"
              style="margin-bottom: 10px"
            />
            <el-input v-model="editContent" type="textarea" :rows="22" resize="vertical" />
            <div class="edit-actions">
              <el-button @click="editing = false">取消</el-button>
              <el-button type="primary" :loading="editLoading" @click="saveEdit">保存内容</el-button>
            </div>
          </template>

          <template v-else>
            <el-alert
              v-if="detail.pending_rebuild"
              type="warning"
              :closable="false"
              show-icon
              title="内容已修改但尚未重建向量，检索仍是旧内容"
              style="margin-bottom: 10px"
            />
            <el-collapse>
              <el-collapse-item
                v-for="chunk in detail.chunks"
                :key="chunk.chunk_id"
                :title="`#${chunk.chunk_index} ${chunk.heading_path || '(无标题)'} · ${chunk.char_count} 字`"
              >
                <div class="chunk-text">{{ chunk.text }}</div>
              </el-collapse-item>
            </el-collapse>
          </template>
        </template>
      </div>
    </el-drawer>
  </div>
</template>
