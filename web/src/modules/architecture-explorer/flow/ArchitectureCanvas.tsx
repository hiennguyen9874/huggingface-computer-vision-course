import { useEffect, useMemo, useState } from 'react'
import {
  Background,
  BaseEdge,
  Controls,
  MarkerType,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  getSmoothStepPath,
  useNodesState,
  useReactFlow,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeMouseHandler,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { ArchitectureModel, ArchitectureNode } from '../model-types'
import { ArchitectureNodeView } from './ArchitectureNode'
import { layoutArchitecture, type LayoutDirection } from './elk-layout'

const nodeTypes = {
  image: ArchitectureNodeView,
  tensor: ArchitectureNodeView,
  linear: ArchitectureNodeView,
  normalization: ArchitectureNodeView,
  attention: ArchitectureNodeView,
  tensorOp: ArchitectureNodeView,
  fusion: ArchitectureNodeView,
  output: ArchitectureNodeView,
  group: ArchitectureNodeView,
  parameter: ArchitectureNodeView,
}

type EdgeHighlight = 'upstream' | 'downstream' | 'muted' | undefined

function SemanticEdge(props: EdgeProps) {
  const [path, labelX, labelY] = getSmoothStepPath({
    sourceX: props.sourceX,
    sourceY: props.sourceY,
    targetX: props.targetX,
    targetY: props.targetY,
    sourcePosition: props.sourcePosition,
    targetPosition: props.targetPosition,
    borderRadius: 10,
    offset: props.type === 'residual' ? 42 : 28,
  })
  const highlight = (props.data as { highlight?: EdgeHighlight } | undefined)?.highlight
  const classes = [
    props.type === 'parameter' ? 'parameter-edge' : props.type === 'residual' ? 'residual-edge' : 'data-edge',
    highlight ? `edge--${highlight}` : '',
  ].filter(Boolean).join(' ')

  return (
    <>
      <BaseEdge path={path} markerEnd={props.markerEnd} className={classes} />
      {props.label ? <text x={labelX} y={labelY - 8} className="edge-label" textAnchor="middle">{String(props.label)}</text> : null}
    </>
  )
}

const edgeTypes = { data: SemanticEdge, parameter: SemanticEdge, residual: SemanticEdge }

function toFlowNodes(model: ArchitectureModel): Node[] {
  return model.nodes.map((node) => ({
    id: node.id,
    type: node.kind,
    position: { x: 0, y: 0 },
    data: node as unknown as Record<string, unknown>,
  }))
}

function pathSets(model: ArchitectureModel, selectedId: string | null) {
  const upstream = new Set<string>()
  const downstream = new Set<string>()
  if (!selectedId) return { upstream, downstream }

  const walk = (start: string, reverse: boolean, result: Set<string>) => {
    const queue = [start]
    while (queue.length) {
      const current = queue.shift()!
      for (const edge of model.edges) {
        const next = reverse && edge.target === current ? edge.source : !reverse && edge.source === current ? edge.target : null
        if (next && next !== selectedId && !result.has(next)) { result.add(next); queue.push(next) }
      }
    }
  }
  walk(selectedId, true, upstream)
  walk(selectedId, false, downstream)
  return { upstream, downstream }
}

function toFlowEdges(model: ArchitectureModel, selectedId: string | null): Edge[] {
  const { upstream, downstream } = pathSets(model, selectedId)
  return model.edges.map((edge) => {
    let highlight: EdgeHighlight
    if (selectedId) {
      if (upstream.has(edge.source) && (upstream.has(edge.target) || edge.target === selectedId)) highlight = 'upstream'
      else if ((edge.source === selectedId || downstream.has(edge.source)) && downstream.has(edge.target)) highlight = 'downstream'
      else highlight = 'muted'
    }
    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      sourceHandle: edge.sourcePort ?? 'output',
      targetHandle: edge.targetPort ?? 'input',
      type: edge.kind,
      label: edge.label,
      data: { highlight },
      markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
    }
  })
}

type CanvasProps = {
  model: ArchitectureModel
  direction: LayoutDirection
  onSelectNode: (node: ArchitectureNode | null) => void
}

function ArchitectureCanvasInner({ model, direction, onSelectNode }: CanvasProps) {
  const baseNodes = useMemo(() => toFlowNodes(model), [model])
  const layoutEdges = useMemo(() => toFlowEdges(model, null), [model])
  const [nodes, setNodes, onNodesChange] = useNodesState(baseNodes)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [layoutReady, setLayoutReady] = useState(false)
  const { fitView } = useReactFlow()

  useEffect(() => {
    let active = true
    void document.fonts.ready.then(async () => {
      const laidOut = await layoutArchitecture(baseNodes, layoutEdges, direction)
      if (!active) return
      setNodes(laidOut)
      setLayoutReady(true)
      requestAnimationFrame(() => void fitView({ padding: 0.14, duration: 450 }))
    })
    return () => { active = false }
  }, [baseNodes, layoutEdges, direction, fitView, setNodes])

  const paths = useMemo(() => pathSets(model, selectedId), [model, selectedId])
  const visibleNodes = useMemo(() => nodes.map((node) => ({
    ...node,
    className: !selectedId ? '' : node.id === selectedId ? 'path-selected' : paths.upstream.has(node.id) ? 'path-upstream' : paths.downstream.has(node.id) ? 'path-downstream' : 'path-muted',
  })), [nodes, paths, selectedId])
  const edges = useMemo(() => toFlowEdges(model, selectedId), [model, selectedId])

  const handleNodeClick: NodeMouseHandler = (_, flowNode) => {
    setSelectedId(flowNode.id)
    onSelectNode(flowNode.data as unknown as ArchitectureNode)
  }
  const clearSelection = () => { setSelectedId(null); onSelectNode(null) }

  return (
    <div className={`flow-shell${layoutReady ? ' is-ready' : ''}`}>
      <div className="flow-hint">{model.projection === 'overview' ? 'OVERVIEW · switch to Detailed to inspect operators' : 'DETAILED · select a node to trace its data path'}</div>
      <ReactFlow
        nodes={visibleNodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodesChange={onNodesChange}
        onNodeClick={handleNodeClick}
        onPaneClick={clearSelection}
        nodesConnectable={false}
        minZoom={0.2}
        maxZoom={1.7}
        proOptions={{ hideAttribution: true }}
      >
        <Background color="#24304a" gap={28} size={1} />
        <MiniMap pannable zoomable nodeColor={(node) => ({ image: '#50bda1', tensor: '#5d8ef0', linear: '#a879e6', normalization: '#73b8d2', attention: '#e7a64a', tensorOp: '#668fef', fusion: '#d58bd8', group: '#e7a64a', output: '#ec6687', parameter: '#8c96ad' })[node.type ?? 'linear'] ?? '#8c96ad'} maskColor="rgba(9, 14, 27, .72)" />
        <Controls showInteractive={false} />
      </ReactFlow>
      {!layoutReady ? <div className="layout-status">Arranging model graph…</div> : null}
    </div>
  )
}

export function ArchitectureCanvas(props: CanvasProps) {
  return <ReactFlowProvider><ArchitectureCanvasInner {...props} /></ReactFlowProvider>
}
