import type { ArchitectureDefinition, ArchitectureEdge, ArchitectureModel, ArchitectureNode, ArchitectureOptions, ArchitectureProjection } from '../../modules/architecture-explorer'

const common = {
  name: 'OneFormer',
  subtitle: 'One task-conditioned model for semantic, instance, and panoptic segmentation · compact reference contracts',
}

const imageNode: ArchitectureNode = { id: 'image', kind: 'image', stage: 'Inputs', visual: 'image', eyebrow: 'VISION INPUT', label: 'RGB images', description: 'A batch of RGB images. The compact demonstration uses 32 × 32 images.', shape: ['N', 3, 'H', 'W'] }
const taskNode: ArchitectureNode = { id: 'task', kind: 'input', stage: 'Task conditioning', visual: 'tokens', eyebrow: 'TASK INPUT', label: 'Segmentation task', description: 'One integer per image selects semantic, instance, or panoptic behavior.', shape: ['N'], operation: 'task_ids ∈ {0,1,2}', notes: ['The chapter describes this task input as the mechanism that makes one set of weights task-dynamic.'] }

const overviewNodes: ArchitectureNode[] = [
  imageNode,
  { id: 'pixels', kind: 'group', stage: 'Pixel module', visual: 'featureMap', eyebrow: 'IMAGE ENCODER', label: 'Pixel embedding module', description: 'A non-overlapping 4 × 4 projection downsamples the image, then a padded 3 × 3 convolution preserves the D-channel quarter-resolution field.', inputShape: ['N', 3, 'H', 'W'], outputShape: ['N', 'D', 'H/4', 'W/4'], operation: 'Conv4×4,s4 → GELU → Conv3×3,p1', parameters: { 'demo D': 64, stride: 4 } },
  taskNode,
  { id: 'conditioning', kind: 'fusion', stage: 'Task conditioning', eyebrow: 'TASK-GUIDED QUERIES', label: 'Condition every query', description: 'A learned task token is added to every learned base query, changing the decoder input for the requested segmentation task.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: 'base_queries + task_token', ports: [{ id: 'queries', label: 'base queries', direction: 'input' }, { id: 'task', label: 'task token', direction: 'input' }, { id: 'output', label: 'guided queries', direction: 'output' }], shapeDerivation: 'N × Q × D + N × 1 × D broadcasts the task token across Q, producing N × Q × D.', parameters: { tasks: 3, 'demo Q': 6, 'demo D': 64 } },
  { id: 'decoder', kind: 'group', stage: 'Transformer', visual: 'transformer', eyebrow: 'TASK-CONDITIONED DECODER', label: 'Transformer decoder', repetition: 2, description: 'Task-guided queries attend to flattened image features and become visual segment queries.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: 'self-attention → pixel cross-attention → FFN', ports: [{ id: 'queries', label: 'guided queries', direction: 'input' }, { id: 'memory', label: 'pixel memory', direction: 'input' }, { id: 'output', label: 'decoded queries', direction: 'output' }], parameters: { layers: 2, heads: 4, 'head dim': 16, 'FFN width': 256 } },
  { id: 'prediction', kind: 'group', stage: 'Prediction', eyebrow: 'SHARED SEGMENTATION HEADS', label: 'Class + mask prediction', description: 'Shared heads emit K+1 class logits and D-wide mask embeddings; pixel-query dot products produce masks.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'H/4', 'W/4'], operation: 'class Linear ∥ mask Linear → einsum' },
  { id: 'result', kind: 'output', stage: 'Output', eyebrow: 'TASK-SPECIFIC INTERPRETATION', label: 'Universal segmentation', description: 'The same predictions are post-processed according to the task input as semantic, instance, or panoptic segmentation.', outputShape: ['N', 'Q', 'H/4', 'W/4'], notes: ['The chapter example uses processor post-processing to resize predictions and interpret them for the selected task.'] },
]

const overviewEdges: ArchitectureEdge[] = [
  { id: 'image-pixels', source: 'image', target: 'pixels', kind: 'data' },
  { id: 'task-condition', source: 'task', target: 'conditioning', targetPort: 'task', kind: 'parameter' },
  { id: 'condition-decode', source: 'conditioning', target: 'decoder', targetPort: 'queries', kind: 'data' },
  { id: 'pixels-decode', source: 'pixels', target: 'decoder', targetPort: 'memory', kind: 'data', label: 'flattened K/V' },
  { id: 'decode-predict', source: 'decoder', target: 'prediction', kind: 'data' },
  { id: 'pixels-predict', source: 'pixels', target: 'prediction', kind: 'residual', label: 'pixel embeddings' },
  { id: 'predict-result', source: 'prediction', target: 'result', kind: 'data' },
]

const pixelDetail: ArchitectureNode[] = [
  { id: 'patch-conv', kind: 'convolution', stage: 'Pixel module', eyebrow: 'PATCH PROJECTION + DOWNSAMPLE', label: '4 × 4 image projection', description: 'A kernel equal to stride partitions a divisible image into non-overlapping 4 × 4 regions and projects RGB directly to D channels.', inputShape: ['N', 3, 'H', 'W'], outputShape: ['N', 'D', 'H/4', 'W/4'], operation: 'Conv2d(3,D,k=4,s=4,p=0)', formula: String.raw`H_o=\lfloor(H-4)/4\rfloor+1`, shapeDerivation: 'When H and W are divisible by 4, the output grid is H/4 × W/4. For the 32 × 32 demo it is 8 × 8.', parameters: { kernel: '4 × 4', stride: 4, padding: 0, 'demo channels': '3 → 64' } },
  { id: 'gelu', kind: 'activation', stage: 'Pixel module', eyebrow: 'ACTIVATION', label: 'GELU', description: 'Elementwise nonlinearity; shape is preserved.', inputShape: ['N', 'D', 'H/4', 'W/4'], outputShape: ['N', 'D', 'H/4', 'W/4'], operation: 'GELU(x)' },
  { id: 'pixel-conv', kind: 'convolution', stage: 'Pixel module', visual: 'featureMap', eyebrow: 'PIXEL EMBEDDINGS', label: 'Context convolution', description: 'A padded 3 × 3 convolution mixes neighboring patch features without changing resolution or width.', inputShape: ['N', 'D', 'H/4', 'W/4'], outputShape: ['N', 'D', 'H/4', 'W/4'], operation: 'Conv2d(D,D,k=3,s=1,p=1)', parameters: { kernel: '3 × 3', stride: 1, padding: 1 } },
  { id: 'memory', kind: 'tensorOp', stage: 'Pixel module', eyebrow: 'SPATIAL → TOKEN MEMORY', label: 'Flatten pixel memory', description: 'Flatten and transpose the D-channel feature map for decoder cross-attention.', inputShape: ['N', 'D', 'h', 'w'], outputShape: ['N', 'h·w', 'D'], operation: 'flatten(2).transpose(1,2)', shapeDerivation: 'For 32 × 32 input, h·w = 8·8 = 64 memory tokens.' },
]

const conditioningCollapsed: ArchitectureNode = { ...overviewNodes[3], id: 'condition' }
const conditioningExpanded: ArchitectureNode[] = [
  { id: 'task-embedding', kind: 'parameter', stage: 'Task conditioning', visual: 'tokens', eyebrow: 'LEARNED TASK TABLE', label: 'Task embedding', description: 'Lookup one D-dimensional vector for semantic, instance, or panoptic task ID.', inputShape: ['N'], outputShape: ['N', 1, 'D'], operation: 'Embedding(3,D)(task_ids).unsqueeze(1)', parameters: { entries: 3, 'demo width': 64 } },
  { id: 'base-queries', kind: 'parameter', stage: 'Task conditioning', visual: 'tokens', eyebrow: 'LEARNED OBJECT QUERIES', label: 'Base queries', description: 'Q learned query vectors shared by every task and expanded across the batch.', shape: [1, 'Q', 'D'], operation: 'Embedding(Q,D) → expand(N,−1,−1)', parameters: { 'demo Q': 6, 'demo D': 64 } },
  { id: 'guided', kind: 'fusion', stage: 'Task conditioning', eyebrow: 'BROADCAST ADD', label: 'Task-guided queries', description: 'The selected task vector offsets every base query.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: 'base_queries + task_tokens', ports: [{ id: 'queries', label: 'N×Q×D', direction: 'input' }, { id: 'task', label: 'N×1×D', direction: 'input' }, { id: 'output', label: 'guided', direction: 'output' }], shapeDerivation: 'Broadcasting repeats N × 1 × D along the query axis without concatenation.' },
]

const decoderCollapsed: ArchitectureNode = { ...overviewNodes[4], id: 'decode', eyebrow: 'COLLAPSED · EXPAND TO INSPECT' }
const decoderExpanded: ArchitectureNode[] = [
  { id: 'query-self', kind: 'attention', stage: 'Transformer', eyebrow: 'QUERY MIXING', label: 'Query self-attention', repetition: 2, description: 'Task-guided query slots exchange information.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: '4-head self-attention', parameters: { heads: 4, 'head dim': 16, map: 'N × 4 × Q × Q' } },
  { id: 'pixel-attn', kind: 'attention', stage: 'Transformer', eyebrow: 'IMAGE CROSS-ATTENTION', label: 'Pixel cross-attention', repetition: 2, description: 'Queries read the flattened H/4 × W/4 image memory.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: 'Attention(queries, pixels, pixels)', ports: [{ id: 'queries', label: 'Q', direction: 'input' }, { id: 'memory', label: 'K/V', direction: 'input' }, { id: 'output', label: 'context', direction: 'output' }], shapeDerivation: 'Scores are N × 4 × Q × (h·w); merging four 16-dimensional heads returns N × Q × 64.', parameters: { heads: 4, 'head dim': 16, map: 'N × 4 × Q × (h·w)' } },
  { id: 'query-ffn', kind: 'linear', stage: 'Transformer', eyebrow: 'QUERY FFN', label: 'Feed-forward network', repetition: 2, description: 'The PyTorch decoder layer expands each query to 4D and projects back to D.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: 'Linear(D,4D) → ReLU → Linear(4D,D)', parameters: { width: '64 → 256 → 64', dropout: 0 } },
]

const predictionNodes: ArchitectureNode[] = [
  { id: 'class-head', kind: 'linear', stage: 'Prediction', eyebrow: 'CLASS BRANCH', label: 'Class head', description: 'One K+1 class score vector per decoded query; the extra class means no segment.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'K+1'], operation: 'Linear(D,K+1)', parameters: { 'demo K': 3, output: 'K+1 = 4' } },
  { id: 'mask-head', kind: 'linear', stage: 'Prediction', eyebrow: 'MASK BRANCH', label: 'Mask embedding head', description: 'Projects decoded queries into the same D-dimensional space as image pixels.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: 'Linear(D,D)' },
  { id: 'mask-dot', kind: 'fusion', stage: 'Prediction', eyebrow: 'QUERY–PIXEL FUSION', label: 'Mask dot product', description: 'Contract D between each query mask embedding and every pixel embedding.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'h', 'w'], operation: 'einsum("nqd,ndhw→nqhw")', ports: [{ id: 'queries', label: 'mask embeddings', direction: 'input' }, { id: 'pixels', label: 'pixels', direction: 'input' }, { id: 'output', label: 'mask logits', direction: 'output' }], formula: String.raw`M_{nqhw}=\sum_d e_{nqd}p_{ndhw}` },
  { id: 'class-out', kind: 'output', stage: 'Output', eyebrow: 'QUERY CLASSES', label: 'Class logits', description: 'Shared per-query class scores.', shape: ['N', 'Q', 'K+1'] },
  { id: 'mask-out', kind: 'output', stage: 'Output', eyebrow: 'UNIVERSAL MASKS', label: 'Task-conditioned masks', description: 'Quarter-resolution query masks interpreted as semantic, instance, or panoptic output according to the selected task.', shape: ['N', 'Q', 'H/4', 'W/4'] },
]

const contrastiveNodes: ArchitectureNode[] = [
  { id: 'normalize', kind: 'normalization', stage: 'Training objective', eyebrow: 'TRAINING ONLY', label: 'Normalize visual queries', description: 'L2-normalize decoded query embeddings. The loss then mean-pools Q for the compact demonstration.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: 'F.normalize(decoded, dim=−1)' },
  { id: 'text', kind: 'parameter', stage: 'Training objective', visual: 'tokens', eyebrow: 'MATCHING TEXT / TASK EMBEDDING', label: 'Text embeddings', description: 'A matching D-dimensional text representation per image. The compact script reuses task embeddings as its demonstration text vectors.', shape: ['N', 'D'], notes: ['The chapter describes query-text contrastive learning; a full OneFormer text mapper is outside the compact script.'] },
  { id: 'contrastive', kind: 'output', stage: 'Training objective', eyebrow: 'AUXILIARY LOSS', label: 'Query–text contrastive loss', description: 'A symmetric image-to-text and text-to-image cross-entropy loss aligns matching diagonal pairs.', inputShape: ['N', 'N'], outputShape: [], operation: '½[CE(S,diag)+CE(Sᵀ,diag)]', ports: [{ id: 'visual', label: 'visual queries', direction: 'input' }, { id: 'text', label: 'text', direction: 'input' }], formula: String.raw`S=\frac{\operatorname{norm}(\operatorname{mean}_Q V)\operatorname{norm}(T)^T}{\tau}`, parameters: { temperature: 0.1, positives: 'diagonal pairs' } },
]

function detailedModel(options: ArchitectureOptions): ArchitectureModel {
  const expandConditioning = options.expandConditioning ?? false
  const expandDecoder = options.expandDecoder ?? false
  const showContrastive = options.showContrastive ?? true
  const conditionNodes = expandConditioning ? conditioningExpanded : [conditioningCollapsed]
  const conditionExit = expandConditioning ? 'guided' : 'condition'
  const decoderNodes = expandDecoder ? decoderExpanded : [decoderCollapsed]
  const decoderEntry = expandDecoder ? 'query-self' : 'decode'
  const decoderExit = expandDecoder ? 'query-ffn' : 'decode'
  const edges: ArchitectureEdge[] = [
    { id: 'i-patch', source: 'image', target: 'patch-conv', kind: 'data' }, { id: 'patch-gelu', source: 'patch-conv', target: 'gelu', kind: 'data' },
    { id: 'gelu-pixel', source: 'gelu', target: 'pixel-conv', kind: 'data' }, { id: 'pixel-memory', source: 'pixel-conv', target: 'memory', kind: 'data' },
    { id: 'condition-decoder', source: conditionExit, target: decoderEntry, targetPort: expandDecoder ? 'input' : 'queries', kind: 'data' },
    { id: 'decoder-class', source: decoderExit, target: 'class-head', kind: 'data' }, { id: 'decoder-mask', source: decoderExit, target: 'mask-head', kind: 'data' },
    { id: 'mask-dot-q', source: 'mask-head', target: 'mask-dot', targetPort: 'queries', kind: 'data' }, { id: 'mask-dot-pixel', source: 'pixel-conv', target: 'mask-dot', targetPort: 'pixels', kind: 'residual', label: 'N×D×h×w' },
    { id: 'class-output', source: 'class-head', target: 'class-out', kind: 'data' }, { id: 'mask-output', source: 'mask-dot', target: 'mask-out', kind: 'data' },
  ]
  if (expandConditioning) edges.push(
    { id: 'task-embed', source: 'task', target: 'task-embedding', kind: 'parameter' }, { id: 'task-guided', source: 'task-embedding', target: 'guided', targetPort: 'task', kind: 'parameter' },
    { id: 'base-guided', source: 'base-queries', target: 'guided', targetPort: 'queries', kind: 'parameter' },
  )
  else edges.push({ id: 'task-condition', source: 'task', target: 'condition', targetPort: 'task', kind: 'parameter' })
  if (expandDecoder) edges.push(
    { id: 'self-cross', source: 'query-self', target: 'pixel-attn', targetPort: 'queries', kind: 'data' }, { id: 'memory-cross', source: 'memory', target: 'pixel-attn', targetPort: 'memory', kind: 'data' },
    { id: 'cross-ffn', source: 'pixel-attn', target: 'query-ffn', kind: 'data' },
  )
  else edges.push({ id: 'memory-decode', source: 'memory', target: 'decode', targetPort: 'memory', kind: 'data' })
  if (showContrastive) edges.push(
    { id: 'decode-normalize', source: decoderExit, target: 'normalize', kind: 'auxiliary', label: 'training only' },
    { id: 'normalize-loss', source: 'normalize', target: 'contrastive', targetPort: 'visual', kind: 'auxiliary' },
    { id: 'text-loss', source: 'text', target: 'contrastive', targetPort: 'text', kind: 'auxiliary' },
  )
  return { ...common, projection: 'detail', nodes: [imageNode, ...pixelDetail, taskNode, ...conditionNodes, ...decoderNodes, ...predictionNodes, ...(showContrastive ? contrastiveNodes : [])], edges }
}

export const oneformerArchitecture: ArchitectureDefinition = {
  id: 'tiny-oneformer', badge: 'ONEFORMER', context: 'MODEL ATLAS / UNIT 3',
  metrics: [{ value: '3', label: 'TASKS' }, { value: '6', label: 'QUERIES' }, { value: '64', label: 'D_MODEL' }, { value: '4', label: 'HEADS' }, { value: '2', label: 'LAYERS' }],
  stages: [{ label: 'Inputs', color: '#4fc0a2' }, { label: 'Pixel module', color: '#53b9c9' }, { label: 'Task conditioning', color: '#d58bd8' }, { label: 'Transformer', color: '#e5a54a' }, { label: 'Prediction', color: '#a879e6' }, { label: 'Output', color: '#ed6a8c' }],
  toggles: [
    { id: 'expandConditioning', label: 'Expand conditioning', activeLabel: 'Collapse conditioning', projection: 'detail' },
    { id: 'expandDecoder', label: 'Expand decoder', activeLabel: 'Collapse decoder', projection: 'detail' },
    { id: 'showContrastive', label: 'Show training branch', activeLabel: 'Hide training branch', projection: 'detail', defaultValue: true },
  ],
  sources: ['codes/unit03/oneformer.py', 'Unit 3: oneformer.mdx'],
  footerNote: 'Compact model: Q=6 · D=64 · K=3 · tasks={semantic, instance, panoptic}; full-paper backbone/text mapper dimensions are intentionally not assumed',
  createModel: (projection: ArchitectureProjection, options) => projection === 'overview' ? { ...common, projection, nodes: overviewNodes, edges: overviewEdges } : detailedModel(options),
}
