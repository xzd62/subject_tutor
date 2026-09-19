<script setup lang="ts">
import { onMounted, reactive, ref } from 'vue'
import { ElMessage } from 'element-plus'
import { http } from '../api/http'
import type { ModelPreset, ModelSettings } from '../api/types'

const settings = reactive<ModelSettings>({
  model: 'deepseek:deepseek-flash',
  temperature: 0.3,
  max_tokens: 8192,
  timeout_s: 60,
  max_retries: 2,
  api_key: '',
  model_ready: false,
  model_error: null,
})
const presets = ref<ModelPreset[]>([])
const modelOptions = ref<string[]>([])
const saving = ref(false)
const testing = ref(false)

async function load(): Promise<void> {
  const data = await http.get<{ settings: ModelSettings; presets: ModelPreset[] }>('/api/settings')
  Object.assign(settings, data.settings)
  presets.value = data.presets
  modelOptions.value = data.presets.flatMap((preset) => preset.models)
}

async function save(): Promise<void> {
  saving.value = true
  try {
    const data = await http.put<{ settings: ModelSettings }>('/api/settings', {
      model: settings.model,
      temperature: settings.temperature,
      max_tokens: settings.max_tokens,
      timeout_s: settings.timeout_s,
      max_retries: settings.max_retries,
      api_key: settings.api_key,
    })
    Object.assign(settings, data.settings)
    ElMessage.success('设置已保存并生效')
  } catch (error) {
    ElMessage.error((error as Error).message)
  } finally {
    saving.value = false
  }
}

async function testConnection(): Promise<void> {
  testing.value = true
  try {
    const data = await http.post<{ latency_s: number; reply: string }>('/api/settings/test')
    ElMessage.success(`连接正常（${data.latency_s}s）：${data.reply || 'ok'}`)
  } catch (error) {
    ElMessage.error((error as Error).message)
  } finally {
    testing.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="page">
    <h2>模型设置</h2>
    <el-card shadow="never" style="max-width: 720px">
      <el-alert
        v-if="settings.model_error"
        :title="`当前模型不可用：${settings.model_error}`"
        type="warning"
        :closable="false"
        show-icon
        style="margin-bottom: 16px"
      />

      <el-form label-width="120px">
        <el-form-item label="预设厂商">
          <el-select
            :model-value="settings.model"
            filterable
            allow-create
            default-first-option
            style="width: 100%"
            placeholder="选择预设模型或输入自定义 provider:model"
            @update:model-value="(value: string) => (settings.model = value)"
          >
            <el-option-group
              v-for="preset in presets"
              :key="preset.provider"
              :label="`${preset.label}${preset.installed ? '' : '（需安装集成包）'}`"
            >
              <el-option
                v-for="model in preset.models"
                :key="model"
                :label="model"
                :value="model"
              />
            </el-option-group>
          </el-select>
        </el-form-item>

        <el-form-item label="API Key">
          <el-input
            v-model="settings.api_key"
            type="password"
            show-password
            placeholder="仅保存在本机 outputs/web-settings.json"
          />
        </el-form-item>

        <el-form-item label="温度 temperature">
          <el-slider v-model="settings.temperature" :min="0" :max="2" :step="0.1" show-input />
        </el-form-item>

        <el-form-item label="最大 tokens">
          <el-input-number v-model="settings.max_tokens" :min="256" :max="32768" :step="256" />
          <span class="hint">推理模型会先消耗思考 token，建议不低于 2048</span>
        </el-form-item>

        <el-form-item label="超时（秒）">
          <el-input-number v-model="settings.timeout_s" :min="5" :max="600" :step="5" />
        </el-form-item>

        <el-form-item label="失败重试次数">
          <el-input-number v-model="settings.max_retries" :min="0" :max="5" />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" :loading="saving" @click="save">保存并生效</el-button>
          <el-button :loading="testing" @click="testConnection">测试连接</el-button>
        </el-form-item>
      </el-form>

      <el-divider />
      <div class="hint">
        换模型只需改这里：不同厂商需安装对应 langchain-* 集成包（如 langchain-openai）；
        对话历史保留在服务进程内，切换模型不影响当前会话。
      </div>
    </el-card>
  </div>
</template>
