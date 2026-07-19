import ELK, { type ElkExtendedEdge, type ElkNode } from 'elkjs/lib/elk.bundled.js'
import { Position, type Edge, type Node } from '@xyflow/react'

const elk = new ELK()

export type LayoutDirection = 'RIGHT' | 'DOWN'

const nodeSizes: Record<string, { width: number; height: number }> = {
  image: { width: 220, height: 250 },
  input: { width: 230, height: 215 },
  tensor: { width: 230, height: 215 },
  convolution: { width: 280, height: 260 },
  linear: { width: 280, height: 260 },
  normalization: { width: 245, height: 220 },
  activation: { width: 230, height: 205 },
  attention: { width: 330, height: 290 },
  tensorOp: { width: 280, height: 235 },
  fusion: { width: 260, height: 225 },
  group: { width: 390, height: 320 },
  output: { width: 230, height: 240 },
  parameter: { width: 210, height: 185 },
}

export async function layoutArchitecture(nodes: Node[], edges: Edge[], direction: LayoutDirection) {
  const horizontal = direction === 'RIGHT'
  const inputSide = horizontal ? 'WEST' : 'NORTH'
  const outputSide = horizontal ? 'EAST' : 'SOUTH'

  const graph: ElkNode = {
    id: 'architecture',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': direction,
      'elk.edgeRouting': 'ORTHOGONAL',
      'elk.spacing.nodeNode': '54',
      'elk.layered.spacing.nodeNodeBetweenLayers': '104',
      'elk.spacing.edgeNode': '26',
      'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
      'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES',
    },
    children: nodes.map((node) => {
      const size = nodeSizes[node.type ?? 'linear'] ?? nodeSizes.linear
      const architectureNode = node.data as { ports?: Array<{ id: string; direction: 'input' | 'output' }> }
      const ports = architectureNode.ports ?? [
        { id: 'input', direction: 'input' as const },
        { id: 'output', direction: 'output' as const },
      ]
      return {
        id: node.id,
        width: size.width,
        height: size.height,
        layoutOptions: { 'elk.portConstraints': ports.length > 2 ? 'FIXED_ORDER' : 'FIXED_SIDE' },
        ports: ports.map((port, index) => ({
          id: `${node.id}:${port.id}`,
          width: 8,
          height: 8,
          layoutOptions: {
            'elk.port.side': port.direction === 'input' ? inputSide : outputSide,
            'elk.port.index': String(index),
          },
        })),
      }
    }),
    edges: edges.map((edge): ElkExtendedEdge => ({
      id: edge.id,
      sources: [`${edge.source}:${edge.sourceHandle ?? 'output'}`],
      targets: [`${edge.target}:${edge.targetHandle ?? 'input'}`],
    })),
  }

  const result = await elk.layout(graph)
  const positions = new Map(result.children?.map((child) => [child.id, { x: child.x ?? 0, y: child.y ?? 0 }]))

  return nodes.map((node) => ({
    ...node,
    position: positions.get(node.id) ?? node.position,
    sourcePosition: horizontal ? Position.Right : Position.Bottom,
    targetPosition: horizontal ? Position.Left : Position.Top,
  }))
}
