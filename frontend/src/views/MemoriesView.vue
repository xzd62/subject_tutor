<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Delete, Edit, Plus } from '@element-plus/icons-vue'
import { http } from '../api/http'
import type { MemoryDetail, MemoryItem } from '../api/types'

const items = ref<MemoryItem[]>([])
const types = ref<string[]>([])
const dialogVisible = ref(false)
const saving = ref(false)
const editingName = ref('')
const form = reactive({ name: '', type: '', description: '', content: '' })

async function load(): Promise<void> {
  const data = await http.get<{ items: MemoryItem[]; types: string[] }>('/api/memories')
  items.value = data.items
  types.value = data.types
}

function openCreate(): void {
  editingName.value = ''
  form.name = ''
  form.type = types.value[0] ?? ''
  form.description = ''
  form.content = ''
  dialogVisible.value = true
}

async function openEdit(row: MemoryItem): Promise<void> {
  try {
    const data = await http.get<MemoryDetail>(
      `/api/memories/${encodeURIComponent(row.name)}`,
    )
    editingName.value = data.name
    form.name = data.name
    form.type = data.type || types.value[0] || ''
    form.description = data.description
    form.content = data.body
    dialogVisible.value = true
  } catch (error) {
    ElMessage.error((error as Error).message)
  }
}

async function save(): Promise<void> {
  if (!form.name.trim() || !form.content.trim()) {
    ElMessage.warning('记忆名称和正文不能为空')
    return
  }
  saving.value = true
  try {
    if (editingName.value) {
      await http.put(`/api/memories/${encodeURIComponent(editingName.value)}`, {
        type: form.type,
        description: form.description,
        content: form.content,
      })
      ElMessage.success('记忆已更新')
    } else {
      await http.post('/api/memories', {
        name: form.name.trim(),
        type: form.type,
        description: form.description,
        content: form.content,
      })
      ElMessage.success('记忆已创建')
    }
    dialogVisible.value = false
    await load()
  } catch (error) {
    ElMessage.error((error as Error).message)
  } finally {
    saving.value = false
  }
}

async function remove(row: MemoryItem): Promise<void> {
  try {
    await ElMessageBox.confirm(`确定删除记忆「${row.name}」？删除后 AI 将不再记得这些内容。`, '删除记忆', {
      type: 'warning',
      confirmButtonText: '删除',
      cancelButtonText: '取消',
    })
  } catch {
    return
  }
  try {
    await http.del(`/api/memories/${encodeURIComponent(row.name)}`)
    ElMessage.success('已删除')
    await load()
  } catch (error) {
    ElMessage.error((error as Error).message)
  }
}

onMounted(load)
</script>

<template>
  <div class="page">
    <h2>记忆管理</h2>
    <el-card shadow="never">
      <template #header>
        <div class="card-header">
          <span>AI 的长期记忆</span>
          <div class="table-actions">
            <span class="muted">共 {{ items.length }} 条，会跨对话生效</span>
            <el-button size="small" type="primary" :icon="Plus" @click="openCreate">
              新建记忆
            </el-button>
          </div>
        </div>
      </template>
      <el-table :data="items" size="small" empty-text="还没有记忆，AI 会在对话中自动记录，也可以手动新建">
        <el-table-column label="名称" min-width="150">
          <template #default="{ row }">
            <span>{{ row.name }}</span>
            <el-tag v-if="row.damaged" type="danger" size="small" class="pending-tag">
              文件损坏
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="type" label="类别" width="110" />
        <el-table-column prop="description" label="描述" min-width="240" show-overflow-tooltip />
        <el-table-column prop="updated_at" label="更新时间" width="170" />
        <el-table-column label="操作" width="150" fixed="right">
          <template #default="{ row }">
            <el-button link type="primary" :icon="Edit" @click="openEdit(row)">编辑</el-button>
            <el-button link type="danger" :icon="Delete" @click="remove(row)">删除</el-button>
          </template>
        </el-table-column>
      </el-table>
    </el-card>

    <el-dialog
      v-model="dialogVisible"
      :title="editingName ? `编辑记忆 · ${editingName}` : '新建记忆'"
      width="640px"
    >
      <el-form label-width="90px">
        <el-form-item label="名称">
          <el-input
            v-model="form.name"
            :disabled="Boolean(editingName)"
            placeholder="如：学生画像、数学错题本"
          />
        </el-form-item>
        <el-form-item label="类别">
          <el-select v-model="form.type" style="width: 220px">
            <el-option v-for="item in types" :key="item" :label="item" :value="item" />
          </el-select>
        </el-form-item>
        <el-form-item label="描述">
          <el-input v-model="form.description" placeholder="一句话说明这条记忆的内容（会出现在 AI 的记忆索引里）" />
        </el-form-item>
        <el-form-item label="正文">
          <el-input v-model="form.content" type="textarea" :rows="14" resize="vertical" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="saving" @click="save">保存</el-button>
      </template>
    </el-dialog>
  </div>
</template>
