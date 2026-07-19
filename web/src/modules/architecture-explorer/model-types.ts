export type TensorShape = Array<number | string>

export type NodeKind =
  | 'image'
  | 'input'
  | 'tensor'
  | 'linear'
  | 'normalization'
  | 'attention'
  | 'tensorOp'
  | 'fusion'
  | 'output'
  | 'group'
  | 'parameter'

export type ArchitecturePort = {
  id: string
  label?: string
  direction: 'input' | 'output'
}

export type ArchitectureNode = {
  id: string
  kind: NodeKind
  label: string
  stage: string
  visual?: 'image' | 'tokens' | 'transformer'
  eyebrow: string
  description: string
  shape?: TensorShape
  inputShape?: TensorShape
  outputShape?: TensorShape
  operation?: string
  formula?: string
  shapeDerivation?: string
  parameters?: Record<string, string | number>
  notes?: string[]
  repetition?: number | string
  ports?: ArchitecturePort[]
}

export type ArchitectureEdge = {
  id: string
  source: string
  target: string
  kind: 'data' | 'parameter' | 'residual'
  label?: string
  sourcePort?: string
  targetPort?: string
}

export type ArchitectureProjection = 'overview' | 'detail'
export type ArchitectureOptions = Record<string, boolean>

export type ArchitectureModel = {
  name: string
  subtitle: string
  projection: ArchitectureProjection
  nodes: ArchitectureNode[]
  edges: ArchitectureEdge[]
}

export type ArchitectureDefinition = {
  id: string
  badge: string
  context: string
  metrics: Array<{ label: string; value: string }>
  stages: Array<{ label: string; color: string }>
  toggles?: Array<{
    id: string
    label: string
    activeLabel: string
    projection?: ArchitectureProjection
    defaultValue?: boolean
  }>
  sources: string[]
  footerNote: string
  createModel: (projection: ArchitectureProjection, options: ArchitectureOptions) => ArchitectureModel
}
