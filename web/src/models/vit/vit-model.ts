import type { ArchitectureDefinition, ArchitectureEdge, ArchitectureModel, ArchitectureNode, ArchitectureProjection } from '../../modules/architecture-explorer/model-types'

const common = {
  name: 'Vision Transformer',
  subtitle: 'Small pre-norm ViT classifier · exact architecture from codes/unit03/vit.py',
}

const overviewNodes: ArchitectureNode[] = [
  {
    id: 'overview-image', kind: 'image', stage: 'Patchify', eyebrow: 'MODEL INPUT', label: 'Normalized RGB images',
    description: 'A batch of CIFAR-sized, square RGB images. All tensors in the reference implementation are float32.',
    shape: ['N', 3, 32, 32], notes: ['N is the symbolic batch size.'],
  },
  {
    id: 'overview-patches', kind: 'group', stage: 'Patchify', visual: 'tokens', eyebrow: 'INPUT PROCESSING', label: 'Patch embedding',
    description: 'A stride-4 convolution projects non-overlapping 4 × 4 patches, then the 8 × 8 grid is flattened into tokens.',
    inputShape: ['N', 3, 32, 32], outputShape: ['N', 64, 64], operation: 'Conv2d → flatten → transpose',
    formula: 'L=(H/P)(W/P)=(32/4)^2=64', shapeDerivation: 'Conv2d spatial output: (32 − 4) / 4 + 1 = 8. The 8 × 8 grid gives L = 64 patch tokens; each has E = 64 features.',
    parameters: { patch: '4 × 4', stride: 4, channels: '3 → 64' },
  },
  {
    id: 'overview-tokenize', kind: 'fusion', stage: 'Tokenize', eyebrow: 'TOKEN ASSEMBLY', label: 'Add class & position',
    description: 'Prepend one learned class token, then add a learned position vector to every sequence position.',
    inputShape: ['N', 64, 64], outputShape: ['N', 65, 64], operation: 'concat([CLS], patches) + position',
    shapeDerivation: '64 patch positions + 1 class position = 65 tokens. Position addition preserves N × 65 × 64.',
  },
  {
    id: 'overview-encoder', kind: 'group', stage: 'Encode', visual: 'transformer', eyebrow: 'BACKBONE', label: 'Transformer encoder',
    description: 'Two pre-norm encoder layers perform global token mixing and channel-wise MLP transformation, each with residual paths.',
    inputShape: ['N', 65, 64], outputShape: ['N', 65, 64], operation: 'Pre-norm MHSA → MLP', repetition: 2,
    formula: '\\operatorname{softmax}\\left(QK^T/\\sqrt{16}\\right)V',
    parameters: { layers: 2, heads: 4, 'head dim': 16, 'MLP width': 256, dropout: 0 },
    notes: ['Select Detailed view to inspect one representative repeated layer.'],
  },
  {
    id: 'overview-readout', kind: 'normalization', stage: 'Classify', eyebrow: 'READOUT', label: 'Class-token readout',
    description: 'Select sequence position zero and apply the final LayerNorm.',
    inputShape: ['N', 65, 64], outputShape: ['N', 64], operation: 'encoded[:, 0] → LayerNorm(64)',
    shapeDerivation: 'Indexing token 0 removes the sequence axis: N × 65 × 64 → N × 64.',
  },
  {
    id: 'overview-logits', kind: 'output', stage: 'Classify', eyebrow: 'MODEL OUTPUT', label: '10 class logits',
    description: 'The linear classification head emits unnormalized class scores.',
    inputShape: ['N', 64], outputShape: ['N', 10], operation: 'Linear(64, 10)',
    parameters: { weights: '10 × 64', bias: 10 },
  },
]

const overviewEdges: ArchitectureEdge[] = overviewNodes.slice(0, -1).map((node, index) => ({
  id: `overview-${index}`, source: node.id, target: overviewNodes[index + 1].id, kind: 'data',
}))

const detailBase: ArchitectureNode[] = [
  {
    id: 'image', kind: 'image', stage: 'Patchify', eyebrow: 'MODEL INPUT', label: 'RGB image',
    description: 'Normalized square RGB images.', shape: ['N', 3, 32, 32], notes: ['H and W must be divisible by patch size P.'],
  },
  {
    id: 'patch-projection', kind: 'linear', stage: 'Patchify', eyebrow: 'PATCH PROJECTION', label: 'Stride-4 convolution',
    description: 'The convolution is equivalent to flattening each non-overlapping patch and applying one shared linear projection, without copying patches.',
    operation: 'Conv2d(3, 64, kernel=4, stride=4)', inputShape: ['N', 3, 32, 32], outputShape: ['N', 64, 8, 8],
    formula: 'H_{out}=\\lfloor(H-K)/S\\rfloor+1=8', shapeDerivation: '(32 − 4) / 4 + 1 = 8 for both spatial axes; output channels equal embedding width E = 64.',
    parameters: { kernel: '4 × 4', stride: 4, padding: 0, channels: '3 → 64' },
  },
  {
    id: 'patch-tokens', kind: 'tensorOp', stage: 'Patchify', eyebrow: 'RESHAPE', label: 'Patch tokens',
    description: 'Flatten the 8 × 8 feature grid and transpose channels into the final embedding axis.',
    inputShape: ['N', 64, 8, 8], outputShape: ['N', 64, 64], operation: 'flatten(2) → transpose(1, 2)',
    formula: 'L=(H/P)(W/P)=8\\cdot8=64', shapeDerivation: 'N × E × 8 × 8 → N × E × 64 → N × 64 × E.',
  },
  {
    id: 'class-token', kind: 'parameter', stage: 'Tokenize', eyebrow: 'LEARNED PARAMETER', label: '[CLS] token',
    description: 'A learned summary token expanded across the batch without copying parameter storage.', shape: [1, 1, 64], operation: 'expand(N, −1, −1)',
  },
  {
    id: 'token-sequence', kind: 'fusion', stage: 'Tokenize', eyebrow: 'CONCATENATE', label: 'Prepend class token',
    description: 'Concatenate along the token axis, reserving sequence position zero for classification.',
    inputShape: ['N', 64, 64], outputShape: ['N', 65, 64], operation: 'cat((cls, patches), dim=1)', shapeDerivation: 'Token count: 1 + 64 = 65; embedding width remains 64.',
  },
  {
    id: 'position-embedding', kind: 'parameter', stage: 'Tokenize', eyebrow: 'LEARNED PARAMETER', label: 'Position embedding',
    description: 'One learned embedding for the class position and every patch position.', shape: [1, 65, 64],
  },
  {
    id: 'positioned-tokens', kind: 'fusion', stage: 'Tokenize', eyebrow: 'ELEMENTWISE ADD', label: 'Positioned tokens',
    description: 'Add position information; broadcasting expands the leading singleton batch axis.',
    inputShape: ['N', 65, 64], outputShape: ['N', 65, 64], operation: 'tokens + position_embedding', shapeDerivation: 'N × 65 × 64 + 1 × 65 × 64 → N × 65 × 64.',
  },
]

const collapsedEncoder: ArchitectureNode = {
  id: 'encoder', kind: 'group', stage: 'Encode', visual: 'transformer', eyebrow: 'COLLAPSED REPEATED BLOCK', label: 'Encoder layer ×2',
  description: 'Two identical PyTorch TransformerEncoderLayer modules, configured pre-norm. Expand the block to inspect the residual branches and projection dimensions.',
  inputShape: ['N', 65, 64], outputShape: ['N', 65, 64], operation: 'LN → MHSA → add → LN → MLP → add', repetition: 2,
  formula: '\\operatorname{Attention}(Q,K,V)=\\operatorname{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V',
  shapeDerivation: 'Attention and MLP both project back to D = 64, so both residual additions preserve N × 65 × 64.',
  parameters: { heads: 4, 'head dim': '64 / 4 = 16', 'MLP width': '64 × 4 = 256', activation: 'GELU', dropout: 0 },
}

const expandedEncoder: ArchitectureNode[] = [
  {
    id: 'attn-norm', kind: 'normalization', stage: 'Encode', eyebrow: 'ENCODER LAYER ×2 · PRE-NORM', label: 'LayerNorm 1',
    description: 'Normalizes each 64-dimensional token before self-attention.', inputShape: ['N', 65, 64], outputShape: ['N', 65, 64], operation: 'LayerNorm(64)', repetition: 2,
  },
  {
    id: 'attention', kind: 'attention', stage: 'Encode', eyebrow: 'GLOBAL TOKEN MIXING', label: '4-head self-attention',
    description: 'Every token supplies queries, keys, and values. Four heads attend over all 65 sequence positions.',
    inputShape: ['N', 65, 64], outputShape: ['N', 65, 64], operation: 'Q/K/V projections → scaled dot-product attention → output projection',
    formula: '\\operatorname{softmax}\\left(QK^T/\\sqrt{16}\\right)V', shapeDerivation: 'Split D = 64 across 4 heads: d_head = 16. Per-head Q, K, V are N × 4 × 65 × 16; concatenation restores N × 65 × 64.',
    parameters: { heads: 4, 'head dim': 16, 'Q/K/V width': 64, 'attention map': 'N × 4 × 65 × 65' }, repetition: 2,
  },
  {
    id: 'attention-add', kind: 'fusion', stage: 'Encode', eyebrow: 'RESIDUAL ADD', label: 'Attention residual',
    description: 'Add the layer input to the attention output.', inputShape: ['N', 65, 64], outputShape: ['N', 65, 64], operation: 'x + attention(LN(x))', repetition: 2,
  },
  {
    id: 'mlp-norm', kind: 'normalization', stage: 'Encode', eyebrow: 'PRE-NORM', label: 'LayerNorm 2',
    description: 'Normalizes the attention-residual stream before the feed-forward network.', inputShape: ['N', 65, 64], outputShape: ['N', 65, 64], operation: 'LayerNorm(64)', repetition: 2,
  },
  {
    id: 'mlp', kind: 'linear', stage: 'Encode', eyebrow: 'CHANNEL MLP', label: 'Feed-forward network',
    description: 'Expand each token independently to four times the embedding width, apply GELU, then project back.',
    inputShape: ['N', 65, 64], outputShape: ['N', 65, 64], operation: 'Linear(64, 256) → GELU → Linear(256, 64)',
    shapeDerivation: 'N × 65 × 64 → N × 65 × 256 → N × 65 × 64. Sequence length is unchanged.', parameters: { expansion: '64 → 256', reduction: '256 → 64', activation: 'GELU', dropout: 0 }, repetition: 2,
  },
  {
    id: 'mlp-add', kind: 'fusion', stage: 'Encode', eyebrow: 'RESIDUAL ADD · BLOCK OUTPUT', label: 'MLP residual',
    description: 'Add the attention stream to the MLP output. This completes the representative layer; the full encoder executes it twice.',
    inputShape: ['N', 65, 64], outputShape: ['N', 65, 64], operation: 'y + MLP(LN(y))', repetition: 2,
    notes: ['The ×2 badge means this entire representative sequence is executed twice, not that its tensors are concatenated.'],
  },
]

const classifierNodes: ArchitectureNode[] = [
  {
    id: 'class-representation', kind: 'normalization', stage: 'Classify', eyebrow: 'TOKEN 0 + FINAL NORM', label: 'Class representation',
    description: 'Select the class token after the encoder, then normalize its embedding.', operation: 'encoded[:, 0] → LayerNorm(64)',
    inputShape: ['N', 65, 64], outputShape: ['N', 64], shapeDerivation: 'Selecting sequence index 0 removes the 65-token axis.',
  },
  {
    id: 'logits', kind: 'output', stage: 'Classify', eyebrow: 'LINEAR HEAD', label: 'Class logits',
    description: 'Unnormalized prediction scores for ten classes.', inputShape: ['N', 64], outputShape: ['N', 10], operation: 'Linear(64, 10)',
    parameters: { weights: '10 × 64', bias: 10 }, notes: ['num_classes defaults to 10 in vit.py.'],
  },
]

function detailEdges(expanded: boolean): ArchitectureEdge[] {
  const encoderEntry = expanded ? 'attn-norm' : 'encoder'
  const encoderExit = expanded ? 'mlp-add' : 'encoder'
  const edges: ArchitectureEdge[] = [
    { id: 'image-projection', source: 'image', target: 'patch-projection', kind: 'data' },
    { id: 'projection-patches', source: 'patch-projection', target: 'patch-tokens', kind: 'data' },
    { id: 'patches-sequence', source: 'patch-tokens', target: 'token-sequence', kind: 'data' },
    { id: 'cls-sequence', source: 'class-token', target: 'token-sequence', kind: 'parameter', label: 'prepend' },
    { id: 'sequence-positioned', source: 'token-sequence', target: 'positioned-tokens', kind: 'data' },
    { id: 'position-positioned', source: 'position-embedding', target: 'positioned-tokens', kind: 'parameter', label: 'add' },
    { id: 'positioned-encoder', source: 'positioned-tokens', target: encoderEntry, kind: 'data' },
    { id: 'encoder-class', source: encoderExit, target: 'class-representation', kind: 'data' },
    { id: 'class-logits', source: 'class-representation', target: 'logits', kind: 'data' },
  ]
  if (expanded) edges.push(
    { id: 'norm-attention', source: 'attn-norm', target: 'attention', kind: 'data' },
    { id: 'attention-add-main', source: 'attention', target: 'attention-add', kind: 'data' },
    { id: 'attention-residual', source: 'positioned-tokens', target: 'attention-add', kind: 'residual', label: 'residual' },
    { id: 'add-norm', source: 'attention-add', target: 'mlp-norm', kind: 'data' },
    { id: 'norm-mlp', source: 'mlp-norm', target: 'mlp', kind: 'data' },
    { id: 'mlp-add-main', source: 'mlp', target: 'mlp-add', kind: 'data' },
    { id: 'mlp-residual', source: 'attention-add', target: 'mlp-add', kind: 'residual', label: 'residual' },
  )
  return edges
}

export function createVitModel(projection: ArchitectureProjection, encoderExpanded = false): ArchitectureModel {
  if (projection === 'overview') return { ...common, projection, nodes: overviewNodes, edges: overviewEdges }
  return {
    ...common,
    projection,
    nodes: [...detailBase, ...(encoderExpanded ? expandedEncoder : [collapsedEncoder]), ...classifierNodes],
    edges: detailEdges(encoderExpanded),
  }
}

export const vitArchitecture: ArchitectureDefinition = {
  id: 'vit-small-cifar',
  badge: 'ViT',
  context: 'MODEL ATLAS / UNIT 3',
  metrics: [
    { value: '32²', label: 'IMAGE' },
    { value: '4²', label: 'PATCH' },
    { value: '65', label: 'TOKENS' },
    { value: '64', label: 'D_MODEL' },
    { value: '2', label: 'LAYERS' },
  ],
  stages: [
    { label: 'Patchify', color: '#4fc0a2' },
    { label: 'Tokenize', color: '#5e8ff1' },
    { label: 'Encode', color: '#e5a54a' },
    { label: 'Classify', color: '#ed6a8c' },
  ],
  toggles: [
    { id: 'encoderExpanded', label: 'Expand encoder', activeLabel: 'Collapse encoder', projection: 'detail' },
  ],
  sources: ['codes/unit03/vit.py', 'Unit 3 ViT chapter'],
  footerNote: 'N = symbolic batch · all tensors float32 · head dimension 64 / 4 = 16',
  createModel: (projection, options) => createVitModel(projection, options.encoderExpanded ?? false),
}
