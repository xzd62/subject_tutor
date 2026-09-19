import { createRouter, createWebHistory } from 'vue-router'
import BooksView from './views/BooksView.vue'
import ChatView from './views/ChatView.vue'
import MemoriesView from './views/MemoriesView.vue'
import SettingsView from './views/SettingsView.vue'

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
    { path: '/chat', name: 'chat', component: ChatView },
    { path: '/books', name: 'books', component: BooksView },
    { path: '/memories', name: 'memories', component: MemoriesView },
    { path: '/settings', name: 'settings', component: SettingsView },
  ],
})
