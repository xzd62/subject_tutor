<script setup lang="ts">
import { useRoute } from 'vue-router'

const route = useRoute()
</script>

<template>
  <el-container class="app-shell">
    <el-aside width="200px" class="sidebar">
      <div class="brand">初中家教</div>
      <el-menu :default-active="route.path" router class="sidebar-menu">
        <el-menu-item index="/chat">AI 对话</el-menu-item>
        <el-menu-item index="/books">教材管理</el-menu-item>
        <el-menu-item index="/memories">记忆管理</el-menu-item>
        <el-menu-item index="/settings">模型设置</el-menu-item>
      </el-menu>
      <div class="sidebar-footer">RAG · LangGraph</div>
    </el-aside>
    <el-main class="main-area">
      <router-view />
    </el-main>
  </el-container>
</template>
