import { Handle, Position, type NodeProps } from '@xyflow/react'
import katex from 'katex'
import 'katex/dist/katex.min.css'
import type { ArchitectureNode } from '../model-types'

function formatShape(shape?: Array<number | string>) {
  return shape ? shape.join(' × ') : null
}

function MathFormula({ expression }: { expression: string }) {
  const html = katex.renderToString(expression, { throwOnError: false, strict: 'warn', trust: false, output: 'html' })
  return <span className="node-formula" dangerouslySetInnerHTML={{ __html: html }} />
}

function ImageGlyph() {
  return (
    <svg className="image-glyph" viewBox="0 0 112 86" role="img" aria-label="Stacked RGB image tensor">
      <defs><linearGradient id="image-sky" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#b9ddff" /><stop offset="1" stopColor="#e4d6ff" /></linearGradient></defs>
      <rect x="12" y="4" width="92" height="72" rx="7" fill="#8fb3e8" opacity=".25" stroke="#5681bd" />
      <rect x="7" y="9" width="92" height="72" rx="7" fill="#b9e5d0" opacity=".55" stroke="#4f9478" />
      <rect x="2" y="14" width="92" height="68" rx="7" fill="url(#image-sky)" stroke="#6b63a8" />
      <circle cx="72" cy="30" r="8" fill="#fff4ae" /><path d="M3 69 29 45l15 14 13-11 36 29v4H3Z" fill="#527aa0" opacity=".72" /><path d="M3 76 30 58l16 11 15-8 32 17v3H3Z" fill="#467d67" opacity=".72" />
    </svg>
  )
}

function TensorGlyph({ parameter = false }: { parameter?: boolean }) {
  return <div className={`token-glyph${parameter ? ' token-glyph--parameter' : ''}`} aria-hidden="true">{Array.from({ length: parameter ? 3 : 8 }, (_, index) => <span key={index} />)}</div>
}

function OperationGlyph({ kind }: { kind: ArchitectureNode['kind'] }) {
  if (kind === 'attention') return <div className="attention-glyph" aria-hidden="true"><span>Q</span><span>K</span><span>V</span><b>ATTN</b><i>4 heads</i></div>
  if (kind === 'fusion') return <div className="operation-glyph operation-glyph--fusion" aria-hidden="true"><span>＋</span><i>merge</i></div>
  if (kind === 'normalization') return <div className="operation-glyph operation-glyph--norm" aria-hidden="true"><span>μσ</span><i>normalize D</i></div>
  if (kind === 'linear') return <div className="operation-glyph operation-glyph--linear" aria-hidden="true"><span>W·x+b</span><i>projection</i></div>
  if (kind === 'tensorOp') return <div className="operation-glyph operation-glyph--reshape" aria-hidden="true"><span>▦ → ▥</span><i>tensor transform</i></div>
  return null
}

function EncoderDiagram() {
  return <div className="encoder-diagram" aria-label="Pre-norm transformer encoder layer"><span>LN</span><b>→</b><span>4-head<br />attention</span><b>＋</b><span>LN</span><b>→</b><span>MLP</span><b>＋</b><i>two residual paths · repeated block</i></div>
}

export function ArchitectureNodeView({ data, selected, sourcePosition, targetPosition }: NodeProps) {
  const node = data as unknown as ArchitectureNode
  const shape = formatShape(node.shape)
  const inputShape = formatShape(node.inputShape)
  const outputShape = formatShape(node.outputShape)

  return (
    <article className={`architecture-node architecture-node--${node.kind}${selected ? ' is-selected' : ''}`}>
      <Handle id="input" type="target" position={targetPosition ?? Position.Left} />
      <div className="node-topline"><span>{node.eyebrow}</span><span className="node-stage">{node.stage}</span></div>
      <h3>{node.label}{node.repetition ? <em className="node-repeat">×{node.repetition}</em> : null}</h3>
      {node.kind === 'image' || node.visual === 'image' ? <ImageGlyph /> : null}
      {node.kind === 'tensor' || node.kind === 'parameter' || node.visual === 'tokens' ? <TensorGlyph parameter={node.kind === 'parameter'} /> : null}
      {node.visual === 'transformer' ? <EncoderDiagram /> : null}
      <OperationGlyph kind={node.kind} />
      {node.kind === 'output' ? <div className="logit-glyph" aria-hidden="true">{[34, 67, 48, 92, 57, 40, 72, 51, 29, 62].map((height, index) => <span key={index} style={{ height: `${height}%` }} />)}</div> : null}
      {shape ? <code className="shape-badge">{shape}</code> : null}
      {inputShape && outputShape ? <div className="shape-transition"><code>{inputShape}</code><span>→</span><code>{outputShape}</code></div> : null}
      {node.operation ? <p className="node-operation">{node.operation}</p> : null}
      {node.parameters ? <div className="parameter-list">{Object.entries(node.parameters).slice(0, 4).map(([key, value]) => <span key={key}><small>{key}</small><b>{value}</b></span>)}</div> : null}
      {node.formula ? <MathFormula expression={node.formula} /> : null}
      <Handle id="output" type="source" position={sourcePosition ?? Position.Right} />
    </article>
  )
}
