import { useMemo, useState } from 'react'
import katex from 'katex'
import type { ArchitectureDefinition, ArchitectureNode, ArchitectureOptions, ArchitectureProjection } from './model-types'
import { ArchitectureCanvas } from './flow/ArchitectureCanvas'
import type { LayoutDirection } from './flow/elk-layout'
import './ArchitectureExplorer.css'

function formatShape(shape?: Array<number | string>) {
  return shape?.join(' × ')
}

function Formula({ expression }: { expression: string }) {
  const html = katex.renderToString(expression, { throwOnError: false, strict: 'warn', trust: false, output: 'html' })
  return <div className="detail-formula" dangerouslySetInnerHTML={{ __html: html }} />
}

function DetailPanel({ node }: { node: ArchitectureNode | null }) {
  return (
    <aside className="detail-panel">
      <div className="detail-heading">
        <span>COMPONENT INSPECTOR</span>
        <i className={node ? `kind-dot kind-dot--${node.kind}` : 'kind-dot'} />
      </div>
      {node ? (
        <div className="detail-content">
          <p className="detail-stage">{node.stage} / {node.eyebrow}</p>
          <h2>{node.label}{node.repetition ? <span className="repeat-badge">×{node.repetition}</span> : null}</h2>
          <p className="detail-description">{node.description}</p>
          <dl>
            {node.operation ? <><dt>Operation</dt><dd>{node.operation}</dd></> : null}
            {node.shape ? <><dt>Tensor</dt><dd><code>{formatShape(node.shape)}</code></dd></> : null}
            {node.inputShape ? <><dt>Input</dt><dd><code>{formatShape(node.inputShape)}</code></dd></> : null}
            {node.outputShape ? <><dt>Output</dt><dd><code>{formatShape(node.outputShape)}</code></dd></> : null}
          </dl>
          {node.shapeDerivation ? <section className="derivation"><h3>Shape derivation</h3><p>{node.shapeDerivation}</p></section> : null}
          {node.formula ? <section className="formula-section"><h3>Formula</h3><Formula expression={node.formula} /></section> : null}
          {node.parameters ? <div className="detail-parameters"><h3>Configuration</h3>{Object.entries(node.parameters).map(([key, value]) => <div key={key}><span>{key}</span><b>{value}</b></div>)}</div> : null}
          {node.notes?.length ? <section className="detail-notes"><h3>Implementation notes</h3>{node.notes.map((note) => <p key={note}>{note}</p>)}</section> : null}
        </div>
      ) : (
        <div className="empty-detail">
          <div className="cursor-icon">⌁</div>
          <h2>Explore the graph</h2>
          <p>Select a component to inspect its tensor contract. Its upstream path turns blue and downstream path turns pink.</p>
        </div>
      )}
      <div className="legend">
        <h3>Visual language</h3>
        <div className="legend-kinds">
          <span><i className="legend-chip chip--input" />input</span><span><i className="legend-chip chip--op" />operation</span>
          <span><i className="legend-chip chip--attention" />attention</span><span><i className="legend-chip chip--fusion" />add / concat</span>
          <span><i className="legend-chip chip--group" />repeated group</span><span><i className="legend-chip chip--output" />output</span>
        </div>
        <span><i className="legend-line" /> activation / tensor flow</span>
        <span><i className="legend-line legend-line--residual" /> residual path</span>
        <span><i className="legend-line legend-line--dashed" /> learned parameter injection</span>
        <p><code>N × T × D</code> = batch × tokens × embedding</p>
      </div>
    </aside>
  )
}

export function ArchitectureExplorer({ definition }: { definition: ArchitectureDefinition }) {
  const [direction, setDirection] = useState<LayoutDirection>('RIGHT')
  const [projection, setProjection] = useState<ArchitectureProjection>('overview')
  const [options, setOptions] = useState<ArchitectureOptions>(() => Object.fromEntries(definition.toggles?.map((toggle) => [toggle.id, toggle.defaultValue ?? false]) ?? []))
  const [selectedNode, setSelectedNode] = useState<ArchitectureNode | null>(null)
  const model = useMemo(() => definition.createModel(projection, options), [definition, projection, options])

  const changeProjection = (next: ArchitectureProjection) => {
    setProjection(next)
    setSelectedNode(null)
  }

  const toggleOption = (id: string) => {
    setOptions((current) => ({ ...current, [id]: !current[id] }))
    setSelectedNode(null)
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
        <div className="title-block">
          <p>{definition.context}</p>
          <h1>{model.name} <span>{definition.badge}</span></h1>
          <p className="subtitle">{model.subtitle}</p>
        </div>
        <div className="header-metrics">
          {definition.metrics.map((metric) => <div key={metric.label}><strong>{metric.value}</strong><span>{metric.label}</span></div>)}
        </div>
      </header>

      <section className="toolbar" aria-label="Diagram controls">
        <div className="view-switch" role="group" aria-label="Level of detail">
          <button className={projection === 'overview' ? 'active' : ''} onClick={() => changeProjection('overview')}>Overview</button>
          <button className={projection === 'detail' ? 'active' : ''} onClick={() => changeProjection('detail')}>Detailed</button>
          {definition.toggles?.filter((toggle) => !toggle.projection || toggle.projection === projection).map((toggle) => (
            <button key={toggle.id} className={`expand-control${options[toggle.id] ? ' active' : ''}`} onClick={() => toggleOption(toggle.id)}>
              {options[toggle.id] ? toggle.activeLabel : toggle.label}
            </button>
          ))}
        </div>
        <div className="toolbar-right">
          <div className="stage-key">
            {definition.stages.map((stage) => <span key={stage.label}><i className="stage-color" style={{ color: stage.color, background: stage.color }} />{stage.label}</span>)}
          </div>
          <div className="layout-switch" role="group" aria-label="Layout direction">
            <button className={direction === 'RIGHT' ? 'active' : ''} onClick={() => setDirection('RIGHT')} title="Horizontal layout">→</button>
            <button className={direction === 'DOWN' ? 'active' : ''} onClick={() => setDirection('DOWN')} title="Vertical layout">↓</button>
          </div>
        </div>
      </section>

      <section className="workspace">
        <ArchitectureCanvas key={`${projection}-${JSON.stringify(options)}`} model={model} direction={direction} onSelectNode={setSelectedNode} />
        <DetailPanel node={selectedNode} />
      </section>

      <footer>
        <span>Sources: {definition.sources.map((source, index) => <span key={source}>{index ? ' · ' : ''}<code>{source}</code></span>)}</span>
        <span>{definition.footerNote}</span>
      </footer>
    </main>
  )
}
