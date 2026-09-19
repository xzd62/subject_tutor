import katex from 'katex'
import { marked } from 'marked'

interface Segment {
  kind: 'code' | 'math'
  text: string
  block: boolean
}

const CJK_RE = /[\u4e00-\u9fff]/

function escapeHtml(value: string): string {
  return value
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

export function renderMarkdown(source: string): string {
  const segments: Segment[] = []
  const token = `@@SEG${Math.random().toString(36).slice(2)}`
  const stash = (kind: 'code' | 'math', text: string, block: boolean): string => {
    segments.push({ kind, text, block })
    return `${token}${segments.length - 1}@@`
  }

  let text = source ?? ''

  text = text.replace(/```[\s\S]*?```/g, (match) => stash('code', match, true))
  text = text.replace(/`[^`\n]+`/g, (match) => stash('code', match, false))
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, (_match, tex: string) => stash('math', tex, true))
  text = text.replace(/\\\[([\s\S]+?)\\\]/g, (_match, tex: string) => stash('math', tex, true))
  text = text.replace(/\\\(([\s\S]+?)\\\)/g, (_match, tex: string) => stash('math', tex, false))
  text = text.replace(/\$([^$\n]+?)\$/g, (_match, tex: string) => stash('math', tex, false))

  let html = marked.parse(text, { async: false }) as string

  const placeholder = new RegExp(`${token}(\\d+)@@`, 'g')
  html = html.replace(placeholder, (_match, index: string) => {
    return restore(segments[Number(index)])
  })
  return html
}

function restore(segment: Segment | undefined): string {
  if (!segment) return ''
  if (segment.kind === 'code') {
    return marked.parse(segment.text, { async: false }) as string
  }
  const tex = segment.text.trim()
  if (!tex) return ''
  if (CJK_RE.test(tex)) {
    return fallback(segment)
  }
  try {
    return katex.renderToString(tex, {
      displayMode: segment.block,
      throwOnError: true,
      strict: false,
    })
  } catch {
    return fallback(segment)
  }
}

function fallback(segment: Segment): string {
  const text = escapeHtml(segment.text.trim())
  return segment.block
    ? `<pre class="math-fallback">${text}</pre>`
    : `<code class="math-fallback">${text}</code>`
}
