import { describe, expect, it } from 'vitest'
import { renderMarkdown } from './markdown'

describe('renderMarkdown 公式兜底', () => {
  it('渲染行内公式', () => {
    const html = renderMarkdown('公式 $a^2+b^2=c^2$ 结束')
    expect(html).toContain('katex')
    expect(html).not.toContain('math-fallback')
  })

  it('渲染块级公式', () => {
    const html = renderMarkdown('$$\nx = \\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}\n$$')
    expect(html).toContain('katex-display')
  })

  it('支持 \\( \\) 与 \\[ \\] 定界符', () => {
    expect(renderMarkdown('\\(x+1\\)')).toContain('katex')
    expect(renderMarkdown('\\[x+1\\]')).toContain('katex-display')
  })

  it('非法 TeX 回退为代码展示', () => {
    const html = renderMarkdown('$\\frac{1}{$')
    expect(html).toContain('math-fallback')
  })

  it('中文公式回退为代码展示', () => {
    const html = renderMarkdown('$速度=路程/时间$')
    expect(html).toContain('math-fallback')
    expect(html).toContain('速度')
  })

  it('代码块中的 $ 不被当作公式', () => {
    const html = renderMarkdown('```bash\necho $PATH\nexport $HOME\n```')
    expect(html).toContain('$PATH')
    expect(html).not.toContain('katex')
  })

  it('普通 Markdown 正常渲染', () => {
    const html = renderMarkdown('# 标题\n\n**加粗** 与 `行内代码`')
    expect(html).toContain('<h1')
    expect(html).toContain('<strong>加粗</strong>')
    expect(html).toContain('<code>行内代码</code>')
  })
})
