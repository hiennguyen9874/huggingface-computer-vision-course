import type { ArchitectureDefinition, ArchitectureEdge, ArchitectureModel, ArchitectureNode, ArchitectureOptions, ArchitectureProjection } from '../../modules/architecture-explorer'

const common = {
  name: 'DEtection TRansformer',
  subtitle: 'Direct set prediction with CNN features, image memory, and learned object queries · exact TinyDETR tensor contracts',
}

const imageNode: ArchitectureNode = {
  id: 'image', kind: 'image', stage: 'Input', visual: 'image', eyebrow: 'MODEL INPUT', label: 'RGB images',
  description: 'A batch of RGB images. The compact executable lesson demonstrates 64 × 64 inputs.',
  shape: ['B', 3, 64, 64], notes: ['B is the symbolic batch size. The network also accepts other spatial sizes while each positional axis remains at most 32 in this compact implementation.'],
}

const queryNode: ArchitectureNode = {
  id: 'queries', kind: 'parameter', stage: 'Decoder', visual: 'tokens', eyebrow: 'LEARNED OBJECT SLOTS', label: 'Object queries',
  description: 'Ten learned D-dimensional vectors initialize ten parallel prediction slots and are expanded across the batch.',
  shape: [1, 10, 64], outputShape: ['B', 10, 64], operation: 'Embedding(Q,D) → expand(B,−1,−1)',
  parameters: { queries: 10, width: 64 }, notes: ['Queries are learned parameters, not image patches. Unmatched slots are trained as the no-object class.'],
}

const overviewNodes: ArchitectureNode[] = [
  imageNode,
  {
    id: 'backbone', kind: 'group', stage: 'Backbone', visual: 'featureMap', eyebrow: 'CNN FEATURE EXTRACTOR', label: 'Three-stage CNN backbone',
    description: 'Three stride-2 convolutions extract local image features while reducing each spatial axis by eight.',
    inputShape: ['B', 3, 64, 64], outputShape: ['B', 64, 8, 8], operation: '(Conv3×3,s2,p1 → ReLU) ×3', repetition: 3,
    shapeDerivation: 'For each stage: Hout = floor((Hin + 2 − 3)/2) + 1. Therefore 64 → 32 → 16 → 8; channels change 3 → 32 → 64 → 64.',
    parameters: { kernel: '3 × 3', stride: 2, padding: 1 },
  },
  {
    id: 'tokens', kind: 'group', stage: 'Tokenize', visual: 'tokens', eyebrow: 'SPATIAL → SEQUENCE', label: 'Flatten + 2D position',
    description: 'The 8 × 8 feature map becomes 64 image tokens. Learned row and column halves concatenate into one position vector per location and are added to the tokens.',
    inputShape: ['B', 64, 8, 8], outputShape: ['B', 64, 64], operation: 'flatten(2) → transpose(1,2) + position',
    shapeDerivation: 'T = H′W′ = 8·8 = 64. Concatenating 32-wide column and row embeddings gives D = 64.',
  },
  {
    id: 'encoder', kind: 'group', stage: 'Encoder', visual: 'transformer', eyebrow: 'IMAGE MEMORY', label: 'Transformer encoder',
    description: 'Two post-norm encoder layers globally mix all spatial tokens and produce image memory.',
    inputShape: ['B', 64, 64], outputShape: ['B', 64, 64], operation: 'self-attention → add & norm → FFN → add & norm', repetition: 2,
    parameters: { layers: 2, heads: 4, 'head dim': 16, 'FFN width': 256 },
  },
  queryNode,
  {
    id: 'decoder', kind: 'group', stage: 'Decoder', visual: 'transformer', eyebrow: 'QUERY–IMAGE FUSION', label: 'Transformer decoder',
    description: 'Ten object slots first communicate by self-attention, then each slot reads the complete encoded image memory with cross-attention.',
    inputShape: ['B', 10, 64], outputShape: ['B', 10, 64], operation: 'query self-attn → memory cross-attn → FFN', repetition: 2,
    ports: [{ id: 'query', label: 'Q object slots', direction: 'input' }, { id: 'memory', label: 'K/V image memory', direction: 'input' }, { id: 'output', label: 'decoded slots', direction: 'output' }],
    parameters: { layers: 2, heads: 4, 'head dim': 16, 'FFN width': 256 },
  },
  {
    id: 'heads', kind: 'group', stage: 'Prediction', eyebrow: 'PARALLEL PREDICTION HEADS', label: 'Class + box heads',
    description: 'Every decoded slot independently predicts a K+1 class vector and one normalized center-format box.',
    inputShape: ['B', 10, 64], outputShape: ['B', 10, 'K+1 / 4'], operation: 'Linear(D,K+1) ∥ MLP(D,D,4) + sigmoid',
    parameters: { 'demo K': 3, classes: 'K+1 = 4', boxes: 4 },
  },
  {
    id: 'detections', kind: 'output', stage: 'Output', eyebrow: 'DIRECT SET OUTPUT', label: '10 detection candidates',
    description: 'The fixed-size unordered set contains class logits and normalized (center_x, center_y, width, height) boxes. No anchors or NMS are used by the model.',
    outputShape: ['B', 10, '4 classes + 4 box values'], notes: ['The extra class is no-object. At inference, no-object and low-confidence slots are filtered and boxes can be converted to pixel xyxy coordinates.'],
  },
]

const overviewEdges: ArchitectureEdge[] = [
  { id: 'o-image-backbone', source: 'image', target: 'backbone', kind: 'data' },
  { id: 'o-backbone-tokens', source: 'backbone', target: 'tokens', kind: 'data' },
  { id: 'o-tokens-encoder', source: 'tokens', target: 'encoder', kind: 'data' },
  { id: 'o-encoder-decoder', source: 'encoder', target: 'decoder', targetPort: 'memory', kind: 'data', label: 'image memory' },
  { id: 'o-queries-decoder', source: 'queries', target: 'decoder', targetPort: 'query', kind: 'parameter', label: '10 slots' },
  { id: 'o-decoder-heads', source: 'decoder', target: 'heads', kind: 'data' },
  { id: 'o-heads-output', source: 'heads', target: 'detections', kind: 'data' },
]

const collapsedBackbone: ArchitectureNode = { ...overviewNodes[1], id: 'backbone-detail', eyebrow: 'COLLAPSED · EXPAND TO INSPECT' }
const expandedBackbone: ArchitectureNode[] = [
  {
    id: 'conv1', kind: 'convolution', stage: 'Backbone', visual: 'featureMap', eyebrow: 'DOWNSAMPLE 1', label: 'Conv + ReLU', description: 'The first convolution extracts 32 channels and halves both spatial axes.',
    inputShape: ['B', 3, 64, 64], outputShape: ['B', 32, 32, 32], operation: 'Conv2d(3,32,3,2,1) → ReLU', formula: String.raw`H_o=\lfloor(H+2P-K)/S\rfloor+1=32`, parameters: { kernel: 3, stride: 2, padding: 1, channels: '3 → 32' },
  },
  {
    id: 'conv2', kind: 'convolution', stage: 'Backbone', visual: 'featureMap', eyebrow: 'DOWNSAMPLE 2', label: 'Conv + ReLU', description: 'The second convolution doubles channel width to D = 64 and halves the grid again.',
    inputShape: ['B', 32, 32, 32], outputShape: ['B', 64, 16, 16], operation: 'Conv2d(32,64,3,2,1) → ReLU', parameters: { kernel: 3, stride: 2, padding: 1, channels: '32 → 64' },
  },
  {
    id: 'conv3', kind: 'convolution', stage: 'Backbone', visual: 'featureMap', eyebrow: 'DOWNSAMPLE 3', label: 'Conv + ReLU', description: 'The final compact backbone stage preserves D = 64 while producing an 8 × 8 feature field.',
    inputShape: ['B', 64, 16, 16], outputShape: ['B', 64, 8, 8], operation: 'Conv2d(64,64,3,2,1) → ReLU', parameters: { kernel: 3, stride: 2, padding: 1, channels: '64 → 64' },
  },
]

const tokenNodes: ArchitectureNode[] = [
  {
    id: 'flatten', kind: 'tensorOp', stage: 'Tokenize', visual: 'tokens', eyebrow: 'RESHAPE + TRANSPOSE', label: 'Image feature sequence',
    description: 'Flatten only the spatial axes, then move channels to the embedding axis expected by batch-first attention.',
    inputShape: ['B', 64, 8, 8], outputShape: ['B', 64, 64], operation: 'flatten(2).transpose(1,2)', shapeDerivation: 'B × D × H′ × W′ → B × D × (H′W′) → B × (H′W′) × D; here H′W′ = 64.',
  },
  {
    id: 'position', kind: 'parameter', stage: 'Tokenize', visual: 'tokens', eyebrow: 'LEARNED 2D POSITION', label: 'Row + column embeddings',
    description: 'Look up a 32-wide vector for every row and column, broadcast to an 8 × 8 grid, then concatenate the halves.',
    outputShape: ['B', 64, 64], operation: 'cat(column[W′], row[H′], dim=−1) → reshape → expand',
    shapeDerivation: 'At each location: 32 column values + 32 row values = D = 64. The 8 × 8 grid yields 64 position tokens.', parameters: { rows: 'Embedding(32,32)', columns: 'Embedding(32,32)' },
  },
  {
    id: 'positioned', kind: 'fusion', stage: 'Tokenize', eyebrow: 'ELEMENTWISE ADD', label: 'Positioned image tokens',
    description: 'Add image content and matching 2D positions before global attention.', inputShape: ['B', 64, 64], outputShape: ['B', 64, 64], operation: 'sequence + positional_encoding',
    ports: [{ id: 'features', label: 'image tokens', direction: 'input' }, { id: 'position', label: '2D positions', direction: 'input' }, { id: 'output', label: 'positioned tokens', direction: 'output' }],
  },
]

const collapsedEncoder: ArchitectureNode = { ...overviewNodes[3], id: 'encoder-detail', eyebrow: 'COLLAPSED · EXPAND TO INSPECT' }
const expandedEncoder: ArchitectureNode[] = [
  {
    id: 'enc-attn', kind: 'attention', stage: 'Encoder', eyebrow: 'REPRESENTATIVE LAYER ×2', label: 'Image self-attention', description: 'Each of the 64 image tokens attends to every image token.',
    inputShape: ['B', 64, 64], outputShape: ['B', 64, 64], operation: 'MultiheadAttention(D=64, heads=4)', formula: String.raw`\operatorname{softmax}(QK^T/\sqrt{16})V`,
    shapeDerivation: 'D / heads = 64 / 4 = 16. Per-head Q/K/V are B × 4 × 64 × 16; the attention map is B × 4 × 64 × 64.', parameters: { heads: 4, 'head dim': 16, 'attention map': 'B×4×64×64' }, repetition: 2,
  },
  {
    id: 'enc-addnorm1', kind: 'fusion', stage: 'Encoder', eyebrow: 'POST-NORM RESIDUAL', label: 'Attention add + norm', description: 'Add the layer input to self-attention output, then apply LayerNorm.',
    inputShape: ['B', 64, 64], outputShape: ['B', 64, 64], operation: 'LayerNorm(x + self_attention(x))', repetition: 2,
  },
  {
    id: 'enc-ffn', kind: 'linear', stage: 'Encoder', eyebrow: 'TOKEN-WISE FFN', label: 'Feed-forward network', description: 'Independently expand each image token fourfold, apply ReLU, and project back to D.',
    inputShape: ['B', 64, 64], outputShape: ['B', 64, 64], operation: 'Linear(64,256) → ReLU → Linear(256,64)', parameters: { expansion: '64 → 256', reduction: '256 → 64', dropout: 0 }, repetition: 2,
  },
  {
    id: 'enc-addnorm2', kind: 'fusion', stage: 'Encoder', eyebrow: 'BLOCK OUTPUT', label: 'FFN add + norm', description: 'The second residual addition and LayerNorm complete one representative encoder layer.',
    inputShape: ['B', 64, 64], outputShape: ['B', 64, 64], operation: 'LayerNorm(y + FFN(y))', repetition: 2, notes: ['The complete sequence is executed twice. The ×2 tensors are not concatenated.'],
  },
]

const collapsedDecoder: ArchitectureNode = { ...overviewNodes[5], id: 'decoder-detail', eyebrow: 'COLLAPSED · EXPAND TO INSPECT' }
const expandedDecoder: ArchitectureNode[] = [
  {
    id: 'dec-self', kind: 'attention', stage: 'Decoder', eyebrow: 'REPRESENTATIVE LAYER ×2', label: 'Query self-attention', description: 'Object slots exchange evidence and can coordinate a non-duplicated set of predictions.',
    inputShape: ['B', 10, 64], outputShape: ['B', 10, 64], operation: '4-head self-attention', parameters: { heads: 4, 'head dim': 16, 'attention map': 'B×4×10×10' }, repetition: 2,
  },
  {
    id: 'dec-addnorm1', kind: 'fusion', stage: 'Decoder', eyebrow: 'QUERY RESIDUAL', label: 'Self-attention add + norm', description: 'Add the incoming object slots and normalize.', inputShape: ['B', 10, 64], outputShape: ['B', 10, 64], operation: 'LayerNorm(q + self_attention(q))', repetition: 2,
  },
  {
    id: 'cross-attn', kind: 'attention', stage: 'Decoder', eyebrow: 'QUERY–IMAGE FUSION', label: 'Memory cross-attention', description: 'Object slots provide Q while encoded image tokens provide K and V.',
    inputShape: ['B', 10, 64], outputShape: ['B', 10, 64], operation: 'Attention(query slots, image memory, image memory)',
    ports: [{ id: 'query', label: 'Q: B×10×64', direction: 'input' }, { id: 'memory', label: 'K/V: B×64×64', direction: 'input' }, { id: 'output', label: 'query context', direction: 'output' }],
    formula: String.raw`\operatorname{softmax}(QK^T/\sqrt{16})V`, shapeDerivation: 'Scores have shape B × 4 × 10 × 64. Four 16-wide heads merge back to B × 10 × 64.', parameters: { heads: 4, 'head dim': 16, 'attention map': 'B×4×10×64' }, repetition: 2,
  },
  {
    id: 'dec-addnorm2', kind: 'fusion', stage: 'Decoder', eyebrow: 'CROSS-ATTENTION RESIDUAL', label: 'Cross-attention add + norm', description: 'Fuse image context into each query through a residual addition and LayerNorm.', inputShape: ['B', 10, 64], outputShape: ['B', 10, 64], operation: 'LayerNorm(q + cross_attention(q,memory))', repetition: 2,
  },
  {
    id: 'dec-ffn', kind: 'linear', stage: 'Decoder', eyebrow: 'QUERY FFN', label: 'Feed-forward network', description: 'Transform each query independently through the fourfold hidden width.',
    inputShape: ['B', 10, 64], outputShape: ['B', 10, 64], operation: 'Linear(64,256) → ReLU → Linear(256,64)', parameters: { width: '64 → 256 → 64', dropout: 0 }, repetition: 2,
  },
  {
    id: 'dec-addnorm3', kind: 'fusion', stage: 'Decoder', eyebrow: 'BLOCK OUTPUT', label: 'FFN add + norm', description: 'The final residual and normalization produce one embedding for every object slot.',
    inputShape: ['B', 10, 64], outputShape: ['B', 10, 64], operation: 'LayerNorm(z + FFN(z))', repetition: 2,
  },
]

const predictionNodes: ArchitectureNode[] = [
  {
    id: 'class-head', kind: 'linear', stage: 'Prediction', eyebrow: 'CLASS BRANCH', label: 'Class head', description: 'One linear projection emits K object classes plus the no-object class for every query.',
    inputShape: ['B', 10, 64], outputShape: ['B', 10, 4], operation: 'Linear(64, K+1)', shapeDerivation: 'The demo uses K = 3, so K + 1 = 4 logits per query.', parameters: { classes: '3 + no-object', weights: '4 × 64', bias: 4 },
  },
  {
    id: 'box-head', kind: 'linear', stage: 'Prediction', eyebrow: 'BOX BRANCH', label: 'Box MLP + sigmoid', description: 'A two-layer MLP predicts four normalized center-format coordinates per query.',
    inputShape: ['B', 10, 64], outputShape: ['B', 10, 4], operation: 'Linear(64,64) → ReLU → Linear(64,4) → sigmoid', formula: String.raw`(c_x,c_y,w,h)\in[0,1]^4`, parameters: { hidden: 64, output: 4, format: 'cx, cy, w, h' },
  },
  { id: 'class-output', kind: 'output', stage: 'Output', eyebrow: 'SET CLASSIFICATION', label: 'Class logits', description: 'Unnormalized K+1 class scores for each fixed query slot.', shape: ['B', 10, 4] },
  { id: 'box-output', kind: 'output', stage: 'Output', eyebrow: 'SET LOCALIZATION', label: 'Normalized boxes', description: 'Ten center-format boxes in normalized image coordinates.', shape: ['B', 10, 4], formula: String.raw`x_1=(c_x-w/2)W,\quad y_1=(c_y-h/2)H`, notes: ['cxcywh_to_xyxy converts normalized boxes to pixel corner coordinates using the original image width and height.'] },
]

const trainingNodes: ArchitectureNode[] = [
  {
    id: 'matching', kind: 'fusion', stage: 'Training only', eyebrow: 'BIPARTITE MATCHING', label: 'One-to-one assignment', description: 'Match predicted slots to ground-truth objects using class and box similarity before computing set loss.',
    inputShape: ['Q predictions', 'M targets'], outputShape: ['min(Q,M) pairs'], operation: 'Hungarian bipartite matching', notes: ['This behavior is described in the DETR chapter but is not implemented in codes/unit03/detr.py. It affects training, not inference data flow.'],
    ports: [{ id: 'predictions', label: 'predicted set', direction: 'input' }, { id: 'targets', label: 'ground truth', direction: 'input' }, { id: 'output', label: 'matched pairs', direction: 'output' }],
  },
  { id: 'targets', kind: 'input', stage: 'Training only', eyebrow: 'GROUND TRUTH', label: 'Object annotations', description: 'An unordered set of class labels and boxes for each image.', shape: ['B', 'M', 'class + box'] },
  { id: 'set-loss', kind: 'output', stage: 'Training only', eyebrow: 'GLOBAL SET OBJECTIVE', label: 'Matched set loss', description: 'Matched predictions receive class and localization supervision; unmatched query slots learn no-object.', operation: 'classification + box loss', notes: ['Exact loss weights and box-cost formulas are intentionally omitted because the compact source file does not define them.'] },
]

function detailedModel(options: ArchitectureOptions): ArchitectureModel {
  const expandBackbone = options.expandBackbone ?? false
  const expandEncoder = options.expandEncoder ?? false
  const expandDecoder = options.expandDecoder ?? false
  const showTraining = options.showTraining ?? false
  const backbone = expandBackbone ? expandedBackbone : [collapsedBackbone]
  const backboneEntry = expandBackbone ? 'conv1' : 'backbone-detail'
  const backboneExit = expandBackbone ? 'conv3' : 'backbone-detail'
  const encoder = expandEncoder ? expandedEncoder : [collapsedEncoder]
  const encoderEntry = expandEncoder ? 'enc-attn' : 'encoder-detail'
  const encoderExit = expandEncoder ? 'enc-addnorm2' : 'encoder-detail'
  const decoder = expandDecoder ? expandedDecoder : [collapsedDecoder]
  const decoderEntry = expandDecoder ? 'dec-self' : 'decoder-detail'
  const decoderExit = expandDecoder ? 'dec-addnorm3' : 'decoder-detail'
  const edges: ArchitectureEdge[] = [
    { id: 'image-backbone', source: 'image', target: backboneEntry, kind: 'data' },
    { id: 'backbone-flatten', source: backboneExit, target: 'flatten', kind: 'data' },
    { id: 'flatten-positioned', source: 'flatten', target: 'positioned', targetPort: 'features', kind: 'data' },
    { id: 'position-positioned', source: 'position', target: 'positioned', targetPort: 'position', kind: 'parameter', label: 'add' },
    { id: 'positioned-encoder', source: 'positioned', target: encoderEntry, kind: 'data' },
    { id: 'queries-decoder', source: 'queries', target: decoderEntry, targetPort: expandDecoder ? 'input' : 'query', kind: 'parameter' },
    { id: 'decoder-class', source: decoderExit, target: 'class-head', kind: 'data' },
    { id: 'decoder-box', source: decoderExit, target: 'box-head', kind: 'data' },
    { id: 'class-output-edge', source: 'class-head', target: 'class-output', kind: 'data' },
    { id: 'box-output-edge', source: 'box-head', target: 'box-output', kind: 'data' },
  ]
  if (expandBackbone) edges.push(
    { id: 'conv1-conv2', source: 'conv1', target: 'conv2', kind: 'data' },
    { id: 'conv2-conv3', source: 'conv2', target: 'conv3', kind: 'data' },
  )
  if (expandEncoder) edges.push(
    { id: 'enc-attn-main', source: 'enc-attn', target: 'enc-addnorm1', kind: 'data' },
    { id: 'enc-residual1', source: 'positioned', target: 'enc-addnorm1', kind: 'residual', label: 'residual' },
    { id: 'enc-add1-ffn', source: 'enc-addnorm1', target: 'enc-ffn', kind: 'data' },
    { id: 'enc-ffn-main', source: 'enc-ffn', target: 'enc-addnorm2', kind: 'data' },
    { id: 'enc-residual2', source: 'enc-addnorm1', target: 'enc-addnorm2', kind: 'residual', label: 'residual' },
  )
  if (expandDecoder) {
    edges.push(
      { id: 'dec-self-add1', source: 'dec-self', target: 'dec-addnorm1', kind: 'data' },
      { id: 'query-residual', source: 'queries', target: 'dec-addnorm1', kind: 'residual', label: 'residual' },
      { id: 'dec-add1-cross', source: 'dec-addnorm1', target: 'cross-attn', targetPort: 'query', kind: 'data' },
      { id: 'memory-cross', source: encoderExit, target: 'cross-attn', targetPort: 'memory', kind: 'data', label: 'K/V memory' },
      { id: 'cross-add2', source: 'cross-attn', target: 'dec-addnorm2', kind: 'data' },
      { id: 'dec-residual2', source: 'dec-addnorm1', target: 'dec-addnorm2', kind: 'residual', label: 'residual' },
      { id: 'dec-add2-ffn', source: 'dec-addnorm2', target: 'dec-ffn', kind: 'data' },
      { id: 'dec-ffn-add3', source: 'dec-ffn', target: 'dec-addnorm3', kind: 'data' },
      { id: 'dec-residual3', source: 'dec-addnorm2', target: 'dec-addnorm3', kind: 'residual', label: 'residual' },
    )
  } else {
    edges.push({ id: 'memory-decoder', source: encoderExit, target: 'decoder-detail', targetPort: 'memory', kind: 'data', label: 'image memory' })
  }
  if (showTraining) edges.push(
    { id: 'class-matching', source: 'class-output', target: 'matching', targetPort: 'predictions', kind: 'auxiliary' },
    { id: 'box-matching', source: 'box-output', target: 'matching', targetPort: 'predictions', kind: 'auxiliary' },
    { id: 'targets-matching', source: 'targets', target: 'matching', targetPort: 'targets', kind: 'auxiliary' },
    { id: 'matching-loss', source: 'matching', target: 'set-loss', kind: 'auxiliary' },
  )
  return {
    ...common, projection: 'detail',
    nodes: [imageNode, ...backbone, ...tokenNodes, ...encoder, queryNode, ...decoder, ...predictionNodes, ...(showTraining ? trainingNodes : [])], edges,
  }
}

export const detrArchitecture: ArchitectureDefinition = {
  id: 'tiny-detr', badge: 'DETR', context: 'MODEL ATLAS / UNIT 3',
  metrics: [{ value: '64²', label: 'IMAGE' }, { value: '10', label: 'QUERIES' }, { value: '64', label: 'D_MODEL' }, { value: '4', label: 'HEADS' }, { value: '2+2', label: 'ENC+DEC' }],
  stages: [
    { label: 'Input', color: '#4fc0a2' }, { label: 'Backbone', color: '#53b9c9' }, { label: 'Tokenize', color: '#5e8ff1' },
    { label: 'Encoder', color: '#e5a54a' }, { label: 'Decoder', color: '#d58bd8' }, { label: 'Prediction', color: '#a879e6' }, { label: 'Output', color: '#ed6a8c' },
  ],
  toggles: [
    { id: 'expandBackbone', label: 'Expand backbone', activeLabel: 'Collapse backbone', projection: 'detail' },
    { id: 'expandEncoder', label: 'Expand encoder', activeLabel: 'Collapse encoder', projection: 'detail' },
    { id: 'expandDecoder', label: 'Expand decoder', activeLabel: 'Collapse decoder', projection: 'detail' },
    { id: 'showTraining', label: 'Show matching', activeLabel: 'Hide matching', projection: 'detail' },
  ],
  sources: ['codes/unit03/detr.py', 'Unit 3: detr.mdx', 'Unit 3: vision-transformer-for-object-detection.mdx'],
  footerNote: 'Exact compact defaults: B symbolic · 64×64 input · D=64 · Q=10 · K=3 · chapter-only matching is explicitly marked training-only',
  createModel: (projection: ArchitectureProjection, options) => projection === 'overview'
    ? { ...common, projection, nodes: overviewNodes, edges: overviewEdges }
    : detailedModel(options),
}
