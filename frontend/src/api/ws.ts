import type { ChatEvent } from './types'

type Listener = (event: ChatEvent) => void
type Handler = () => void

export class ChatSocket {
  private socket: WebSocket | null = null
  private listeners = new Set<Listener>()
  private openHandlers = new Set<Handler>()
  private closeHandlers = new Set<Handler>()

  get isOpen(): boolean {
    return this.socket?.readyState === WebSocket.OPEN
  }

  connect(): void {
    if (this.socket && this.socket.readyState !== WebSocket.CLOSED) return
    const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
    const socket = new WebSocket(`${protocol}://${location.host}/ws/chat`)
    socket.onopen = () => this.openHandlers.forEach((handler) => handler())
    socket.onmessage = (raw) => {
      let event: ChatEvent
      try {
        event = JSON.parse(raw.data as string) as ChatEvent
      } catch {
        return
      }
      this.listeners.forEach((listener) => listener(event))
    }
    socket.onclose = () => {
      this.socket = null
      this.closeHandlers.forEach((handler) => handler())
    }
    this.socket = socket
  }

  on(listener: Listener): () => void {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  onOpen(handler: Handler): () => void {
    this.openHandlers.add(handler)
    return () => {
      this.openHandlers.delete(handler)
    }
  }

  onClose(handler: Handler): () => void {
    this.closeHandlers.add(handler)
    return () => {
      this.closeHandlers.delete(handler)
    }
  }

  send(payload: unknown): void {
    if (!this.isOpen) {
      throw new Error('连接尚未就绪，请稍候重试')
    }
    this.socket?.send(JSON.stringify(payload))
  }

  close(): void {
    this.socket?.close()
    this.socket = null
  }
}
