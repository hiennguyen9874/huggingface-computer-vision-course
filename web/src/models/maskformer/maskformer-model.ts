import type { ArchitectureDefinition, ArchitectureEdge, ArchitectureModel, ArchitectureNode, ArchitectureOptions, ArchitectureProjection } from '../../modules/architecture-explorer'

const common = {
  name: 'MaskFormer',
  subtitle: 'Pixel embeddings + learned segment queries → class scores and binary masks · TinyMaskFormer reference',
}

const overviewNodes: ArchitectureNode[] = [
  { id: 'image', kind: 'image', stage: 'Input', visual: 'image', eyebrow: 'MODEL INPUT', label: 'RGB images', description: 'A batch of RGB images. The compact script demonstrates 32 × 32 inputs; H and W remain symbolic here.', shape: ['N', 3, 'H', 'W'] },
  { id: 'pixel-module', kind: 'group', stage: 'Pixel module', visual: 'featureMap', eyebrow: 'BACKBONE + PIXEL DECODER', label: 'Pixel-level module', description: 'Three convolutions produce a D-channel embedding for every location on a quarter-resolution grid.', inputShape: ['N', 3, 'H', 'W'], outputShape: ['N', 'D', 'H/4', 'W/4'], operation: 'Conv s2 → ReLU → Conv s2 → ReLU → Conv', shapeDerivation: 'For the demonstrated 32 × 32 input, two stride-2 convolutions produce 16 × 16 then 8 × 8. The final stride-1 convolution preserves 8 × 8.', parameters: { demo: 'D = 64', output: 'N × D × h × w', 'h,w': 'H/4, W/4' } },
  { id: 'sequence', kind: 'tensorOp', stage: 'Transformer', eyebrow: 'SPATIAL → SEQUENCE', label: 'Pixel memory', description: 'Flatten the spatial grid and transpose channels to obtain the transformer memory sequence.', inputShape: ['N', 'D', 'h', 'w'], outputShape: ['N', 'h·w', 'D'], operation: 'flatten(2) → transpose(1, 2)', shapeDerivation: 'N × D × h × w → N × D × (h·w) → N × (h·w) × D.' },
  { id: 'queries', kind: 'parameter', stage: 'Transformer', visual: 'tokens', eyebrow: 'LEARNED PARAMETER', label: 'Segment queries', description: 'Q learned D-dimensional queries are expanded across the batch.', shape: [1, 'Q', 'D'], parameters: { demo: 'Q = 6', width: 'D = 64' } },
  { id: 'decoder', kind: 'group', stage: 'Transformer', visual: 'transformer', eyebrow: 'SEGMENT DECODER', label: 'Transformer decoder', repetition: 2, description: 'Each decoder layer mixes queries with self-attention, reads pixel memory with cross-attention, and applies a feed-forward network.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: 'self-attention → cross-attention(memory) → FFN', ports: [{ id: 'queries', label: 'queries', direction: 'input' }, { id: 'memory', label: 'pixel memory', direction: 'input' }, { id: 'output', label: 'segments', direction: 'output' }], parameters: { layers: 2, heads: 4, 'head dim': 'D/4 = 16', 'FFN width': '4D = 256' } },
  { id: 'heads', kind: 'group', stage: 'Segmentation', eyebrow: 'TWO PREDICTION BRANCHES', label: 'Class + mask heads', description: 'A class projection emits K+1 scores per query; an MLP emits one D-dimensional mask embedding per query.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'K+1'], operation: 'Linear(D,K+1) ∥ MLP(D,D,D)' },
  { id: 'outputs', kind: 'output', stage: 'Output', eyebrow: 'SEGMENT PREDICTIONS', label: 'Classes + binary masks', description: 'Each query predicts a class distribution and one quarter-resolution mask. Masks may be bilinearly upsampled to H × W.', outputShape: ['N', 'Q', 'h', 'w'], operation: 'einsum("nqd,ndhw→nqhw")', formula: 'M_{nqhw}=e_{nq}^{T}p_{nhw}', shapeDerivation: 'The D axis is contracted between N × Q × D mask embeddings and N × D × h × w pixel embeddings, leaving N × Q × h × w. Class logits are N × Q × (K+1).' },
]

const overviewEdges: ArchitectureEdge[] = [
  { id: 'i-p', source: 'image', target: 'pixel-module', kind: 'data' },
  { id: 'p-s', source: 'pixel-module', target: 'sequence', kind: 'data' },
  { id: 's-d', source: 'sequence', target: 'decoder', targetPort: 'memory', kind: 'data', label: 'N×(h·w)×D' },
  { id: 'q-d', source: 'queries', target: 'decoder', targetPort: 'queries', kind: 'parameter' },
  { id: 'd-h', source: 'decoder', target: 'heads', kind: 'data' },
  { id: 'h-o', source: 'heads', target: 'outputs', kind: 'data' },
  { id: 'p-o', source: 'pixel-module', target: 'outputs', kind: 'residual', label: 'pixel embeddings' },
]

const pixelCollapsed: ArchitectureNode = { ...overviewNodes[1], id: 'pixels', eyebrow: 'COLLAPSED PIXEL MODULE' }
const pixelExpanded: ArchitectureNode[] = [
  { id: 'conv1', kind: 'convolution', stage: 'Pixel module', eyebrow: 'DOWNSAMPLE 1', label: '3 → 32 convolution', description: 'The first padded 3 × 3 convolution halves both spatial dimensions.', inputShape: ['N', 3, 'H', 'W'], outputShape: ['N', 32, 'H/2', 'W/2'], operation: 'Conv2d(3,32,k=3,s=2,p=1)', formula: String.raw`H_o=\lfloor(H+2P-K)/S\rfloor+1`, parameters: { kernel: '3 × 3', stride: 2, padding: 1 } },
  { id: 'relu1', kind: 'activation', stage: 'Pixel module', eyebrow: 'ACTIVATION', label: 'ReLU', description: 'Elementwise rectification; tensor dimensions are unchanged.', inputShape: ['N', 32, 'H/2', 'W/2'], outputShape: ['N', 32, 'H/2', 'W/2'], operation: 'max(0, x)' },
  { id: 'conv2', kind: 'convolution', stage: 'Pixel module', eyebrow: 'DOWNSAMPLE 2', label: '32 → D convolution', description: 'A second stride-2 convolution reaches the quarter-resolution pixel grid and expands channels to D.', inputShape: ['N', 32, 'H/2', 'W/2'], outputShape: ['N', 'D', 'H/4', 'W/4'], operation: 'Conv2d(32,D,k=3,s=2,p=1)', parameters: { kernel: '3 × 3', stride: 2, padding: 1, 'demo D': 64 } },
  { id: 'relu2', kind: 'activation', stage: 'Pixel module', eyebrow: 'ACTIVATION', label: 'ReLU', description: 'Elementwise rectification before the final pixel projection.', inputShape: ['N', 'D', 'H/4', 'W/4'], outputShape: ['N', 'D', 'H/4', 'W/4'], operation: 'max(0, x)' },
  { id: 'pixel-embed', kind: 'convolution', stage: 'Pixel module', visual: 'featureMap', eyebrow: 'PIXEL EMBEDDINGS', label: 'D-channel pixel field', description: 'A stride-1 padded convolution mixes local context while preserving channels and resolution.', inputShape: ['N', 'D', 'H/4', 'W/4'], outputShape: ['N', 'D', 'H/4', 'W/4'], operation: 'Conv2d(D,D,k=3,s=1,p=1)', parameters: { kernel: '3 × 3', stride: 1, padding: 1 } },
]

const decoderCollapsed: ArchitectureNode = { ...overviewNodes[4], id: 'decode', eyebrow: 'COLLAPSED · EXPAND TO INSPECT' }
const decoderExpanded: ArchitectureNode[] = [
  { id: 'self-attn', kind: 'attention', stage: 'Transformer', eyebrow: 'DECODER LAYER', label: 'Query self-attention', repetition: 2, description: 'The Q segment slots exchange global information before reading image features.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: '4-head self-attention', parameters: { heads: 4, 'head dim': 16, map: 'N × 4 × Q × Q' } },
  { id: 'cross-attn', kind: 'attention', stage: 'Transformer', eyebrow: 'PIXEL CROSS-ATTENTION', label: 'Queries attend to pixels', repetition: 2, description: 'Segment queries are queries; flattened pixel embeddings provide keys and values.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: 'Attention(query, pixel memory, pixel memory)', ports: [{ id: 'queries', label: 'queries', direction: 'input' }, { id: 'memory', label: 'K/V pixels', direction: 'input' }, { id: 'output', label: 'context', direction: 'output' }], shapeDerivation: 'Attention weights have shape N × 4 × Q × (h·w); concatenating four 16-wide heads restores D=64.', parameters: { heads: 4, 'head dim': 16, map: 'N × 4 × Q × (h·w)' } },
  { id: 'ffn', kind: 'linear', stage: 'Transformer', eyebrow: 'DECODER FFN', label: 'Query feed-forward', repetition: 2, description: 'Each query is independently expanded to 4D and reduced back to D.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: 'Linear(D,4D) → ReLU → Linear(4D,D)', parameters: { 'demo width': '64 → 256 → 64', dropout: 0 } },
]

const tail: ArchitectureNode[] = [
  { id: 'class-head', kind: 'linear', stage: 'Segmentation', eyebrow: 'CLASS BRANCH', label: 'Per-query class head', description: 'Includes one extra no-segment class.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'K+1'], operation: 'Linear(D,K+1)', parameters: { 'demo K': 3, classes: 'K + 1 = 4' } },
  { id: 'mask-head', kind: 'linear', stage: 'Segmentation', eyebrow: 'MASK BRANCH', label: 'Mask embedding MLP', description: 'Projects every decoded segment into the pixel embedding space.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'D'], operation: 'Linear(D,D) → ReLU → Linear(D,D)' },
  { id: 'mask-product', kind: 'fusion', stage: 'Segmentation', eyebrow: 'PIXEL–QUERY FUSION', label: 'Mask dot product', description: 'Contract mask embeddings with the D-channel pixel field.', inputShape: ['N', 'Q', 'D'], outputShape: ['N', 'Q', 'h', 'w'], operation: 'einsum("nqd,ndhw→nqhw")', ports: [{ id: 'masks', label: 'mask embeddings', direction: 'input' }, { id: 'pixels', label: 'pixel embeddings', direction: 'input' }, { id: 'output', label: 'mask logits', direction: 'output' }], formula: String.raw`M_{nqhw}=\sum_{d=1}^{D}e_{nqd}p_{ndhw}` },
  { id: 'class-output', kind: 'output', stage: 'Output', eyebrow: 'CLASS OUTPUT', label: 'Class logits', description: 'Unnormalized K+1 class scores for each segment query.', shape: ['N', 'Q', 'K+1'] },
  { id: 'mask-output', kind: 'output', stage: 'Output', eyebrow: 'MASK OUTPUT', label: 'Mask logits', description: 'One low-resolution binary-mask logit map per query; sigmoid yields probabilities.', shape: ['N', 'Q', 'H/4', 'W/4'], notes: ['The reference demo upsamples 8 × 8 masks to the original 32 × 32 resolution outside the model.'] },
]

function detailModel(options: ArchitectureOptions): ArchitectureModel {
  const expandPixel = options.expandPixel ?? false
  const expandDecoder = options.expandDecoder ?? false
  const pixelNodes = expandPixel ? pixelExpanded : [pixelCollapsed]
  const pixelEntry = expandPixel ? 'conv1' : 'pixels'
  const pixelExit = expandPixel ? 'pixel-embed' : 'pixels'
  const decoderNodes = expandDecoder ? decoderExpanded : [decoderCollapsed]
  const decoderEntry = expandDecoder ? 'self-attn' : 'decode'
  const decoderExit = expandDecoder ? 'ffn' : 'decode'
  const edges: ArchitectureEdge[] = [
    { id: 'image-pixel', source: 'image', target: pixelEntry, kind: 'data' },
    { id: 'pixel-seq', source: pixelExit, target: 'sequence', kind: 'data' },
    { id: 'query-decode', source: 'queries', target: decoderEntry, targetPort: expandDecoder ? 'input' : 'queries', kind: 'parameter' },
    { id: 'decode-class', source: decoderExit, target: 'class-head', kind: 'data' },
    { id: 'decode-mask', source: decoderExit, target: 'mask-head', kind: 'data' },
    { id: 'mask-fusion', source: 'mask-head', target: 'mask-product', targetPort: 'masks', kind: 'data' },
    { id: 'pixels-fusion', source: pixelExit, target: 'mask-product', targetPort: 'pixels', kind: 'residual', label: 'N×D×h×w' },
    { id: 'class-out', source: 'class-head', target: 'class-output', kind: 'data' },
    { id: 'mask-out', source: 'mask-product', target: 'mask-output', kind: 'data' },
  ]
  if (expandPixel) edges.push(
    { id: 'c1-r1', source: 'conv1', target: 'relu1', kind: 'data' }, { id: 'r1-c2', source: 'relu1', target: 'conv2', kind: 'data' },
    { id: 'c2-r2', source: 'conv2', target: 'relu2', kind: 'data' }, { id: 'r2-pe', source: 'relu2', target: 'pixel-embed', kind: 'data' },
  )
  if (expandDecoder) edges.push(
    { id: 'seq-cross', source: 'sequence', target: 'cross-attn', targetPort: 'memory', kind: 'data' },
    { id: 'self-cross', source: 'self-attn', target: 'cross-attn', targetPort: 'queries', kind: 'data' },
    { id: 'cross-ffn', source: 'cross-attn', target: 'ffn', kind: 'data' },
  )
  else edges.push({ id: 'seq-decode', source: 'sequence', target: 'decode', targetPort: 'memory', kind: 'data' })
  return { ...common, projection: 'detail', nodes: [overviewNodes[0], ...pixelNodes, overviewNodes[2], overviewNodes[3], ...decoderNodes, ...tail], edges }
}

export const maskformerArchitecture: ArchitectureDefinition = {
  id: 'tiny-maskformer', badge: 'MASKFORMER', context: 'MODEL ATLAS / UNIT 3',
  metrics: [{ value: 'H/4', label: 'MASK SCALE' }, { value: '6', label: 'QUERIES' }, { value: '64', label: 'D_MODEL' }, { value: '4', label: 'HEADS' }, { value: '2', label: 'LAYERS' }],
  stages: [{ label: 'Input', color: '#4fc0a2' }, { label: 'Pixel module', color: '#53b9c9' }, { label: 'Transformer', color: '#e5a54a' }, { label: 'Segmentation', color: '#d58bd8' }, { label: 'Output', color: '#ed6a8c' }],
  toggles: [{ id: 'expandPixel', label: 'Expand pixel module', activeLabel: 'Collapse pixel module', projection: 'detail' }, { id: 'expandDecoder', label: 'Expand decoder', activeLabel: 'Collapse decoder', projection: 'detail' }],
  sources: ['codes/unit03/maskformer.py', 'Unit 3: vision-transformers-for-image-segmentation.mdx'],
  footerNote: 'Exact compact-model defaults: Q=6 · D=64 · K=3 · h=H/4 · w=W/4 · K+1 includes “no segment”',
  createModel: (projection: ArchitectureProjection, options) => projection === 'overview' ? { ...common, projection, nodes: overviewNodes, edges: overviewEdges } : detailModel(options),
}
