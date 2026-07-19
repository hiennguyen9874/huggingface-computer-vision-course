import type {
  ArchitectureDefinition,
  ArchitectureEdge,
  ArchitectureModel,
  ArchitectureNode,
  ArchitectureOptions,
  ArchitecturePort,
  ArchitectureProjection,
} from '../../modules/architecture-explorer'

const qkvInputs: ArchitecturePort[] = [
  { id: 'q', label: 'Q', direction: 'input' },
  { id: 'k', label: 'K', direction: 'input' },
  { id: 'v', label: 'V', direction: 'input' },
  { id: 'output', label: 'context', direction: 'output' },
]

const qkvOutputs: ArchitecturePort[] = [
  { id: 'input', label: 'X', direction: 'input' },
  { id: 'q', label: 'Q', direction: 'output' },
  { id: 'k', label: 'K', direction: 'output' },
  { id: 'v', label: 'V', direction: 'output' },
]

const inputPort: ArchitecturePort[] = [{ id: 'output', label: 'tokens', direction: 'output' }]
const outputPort: ArchitecturePort[] = [{ id: 'input', label: 'hidden states', direction: 'input' }]

const common = {
  name: 'Encoder–Decoder Transformer',
  subtitle: 'Symbolic reference architecture · pre-norm blocks · dimensions remain explicit where unspecified',
}

const overviewNodes: ArchitectureNode[] = [
  {
    id: 'source', kind: 'input', stage: 'Inputs', visual: 'tokens', eyebrow: 'SOURCE INPUT', label: 'Source token IDs',
    description: 'Tokenized source sequence. Padding positions are excluded with a source padding mask.', shape: ['B', 'S'],
    ports: inputPort, notes: ['B = batch size; S = source sequence length.'],
  },
  {
    id: 'source-embedding', kind: 'linear', stage: 'Inputs', eyebrow: 'SOURCE EMBEDDING', label: 'Token + position embedding',
    description: 'Look up D-dimensional token vectors and add source position information.', inputShape: ['B', 'S'], outputShape: ['B', 'S', 'D'],
    operation: 'Embedding(Vsrc, D) + position(S, D)', shapeDerivation: 'Each of S token IDs maps to one D-dimensional vector; addition preserves B × S × D.',
  },
  {
    id: 'encoder', kind: 'group', stage: 'Encoder', visual: 'transformer', eyebrow: 'REPEATED BACKBONE', label: 'Encoder block', repetition: 'Nₑ',
    description: 'Repeated pre-norm self-attention and feed-forward blocks build contextual source representations.',
    inputShape: ['B', 'S', 'D'], outputShape: ['B', 'S', 'D'], operation: 'LN → self-attention → add → LN → FFN → add',
    formula: String.raw`\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(QK^T/\sqrt{d_h})V`,
    parameters: { layers: 'Nₑ', heads: 'H', 'head dim': 'dₕ = D/H', 'FFN width': 'D_ff' },
  },
  {
    id: 'memory', kind: 'tensor', stage: 'Encoder', visual: 'tokens', eyebrow: 'ENCODER OUTPUT', label: 'Encoder memory',
    description: 'Contextual source states reused as keys and values by every decoder block.', shape: ['B', 'S', 'D'],
  },
  {
    id: 'target', kind: 'input', stage: 'Inputs', visual: 'tokens', eyebrow: 'SHIFTED TARGET INPUT', label: 'Previous target tokens',
    description: 'Teacher-forced targets shifted right during training, or generated tokens during inference.', shape: ['B', 'T'], ports: inputPort,
    notes: ['A causal mask prevents each query position from reading future target positions.'],
  },
  {
    id: 'target-embedding', kind: 'linear', stage: 'Inputs', eyebrow: 'TARGET EMBEDDING', label: 'Token + position embedding',
    description: 'Map target IDs to D-dimensional vectors and add target positions.', inputShape: ['B', 'T'], outputShape: ['B', 'T', 'D'],
    operation: 'Embedding(Vtgt, D) + position(T, D)',
  },
  {
    id: 'decoder', kind: 'group', stage: 'Decoder', visual: 'transformer', eyebrow: 'CONDITIONAL GENERATOR', label: 'Decoder block', repetition: 'N_d',
    description: 'Masked self-attention models the output prefix; cross-attention reads encoder memory; an FFN transforms each position.',
    inputShape: ['B', 'T', 'D'], outputShape: ['B', 'T', 'D'], operation: 'masked self-attn → cross-attn → FFN',
    ports: [
      { id: 'target', label: 'target stream', direction: 'input' },
      { id: 'context', label: 'encoder memory', direction: 'input' },
      { id: 'output', label: 'decoder states', direction: 'output' },
    ],
    parameters: { layers: 'N_d', heads: 'H', 'self map': 'B × H × T × T', 'cross map': 'B × H × T × S' },
  },
  {
    id: 'output', kind: 'output', stage: 'Output', eyebrow: 'SEQUENCE OUTPUT', label: 'Vocabulary logits',
    description: 'A vocabulary projection produces one unnormalized score vector for every target position.',
    inputShape: ['B', 'T', 'D'], outputShape: ['B', 'T', 'Vtgt'], operation: 'Linear(D, Vtgt)', ports: outputPort,
    shapeDerivation: 'The linear layer replaces only the final feature axis D with target vocabulary size Vtgt.',
  },
]

const overviewEdges: ArchitectureEdge[] = [
  { id: 'ov-source-embed', source: 'source', target: 'source-embedding', kind: 'data' },
  { id: 'ov-embed-encoder', source: 'source-embedding', target: 'encoder', kind: 'data' },
  { id: 'ov-encoder-memory', source: 'encoder', target: 'memory', kind: 'data' },
  { id: 'ov-target-embed', source: 'target', target: 'target-embedding', kind: 'data' },
  { id: 'ov-embed-decoder', source: 'target-embedding', target: 'decoder', targetPort: 'target', kind: 'data' },
  { id: 'ov-memory-decoder', source: 'memory', target: 'decoder', targetPort: 'context', kind: 'data', label: 'K, V context' },
  { id: 'ov-decoder-output', source: 'decoder', target: 'output', kind: 'data' },
]

const sourceInputNodes: ArchitectureNode[] = [
  overviewNodes[0],
  { ...overviewNodes[1], id: 'src-embed', label: 'Source embedding' },
]

const targetInputNodes: ArchitectureNode[] = [
  overviewNodes[4],
  { ...overviewNodes[5], id: 'tgt-embed', label: 'Target embedding' },
]

const collapsedEncoder: ArchitectureNode = {
  ...overviewNodes[2], id: 'enc-block', label: 'Encoder block', eyebrow: 'COLLAPSED · EXPAND TO INSPECT',
  notes: ['The complete block is repeated Nₑ times; expanding shows one representative layer.'],
}

const expandedEncoder: ArchitectureNode[] = [
  {
    id: 'enc-ln1', kind: 'normalization', stage: 'Encoder', eyebrow: 'PRE-NORM', label: 'Encoder LayerNorm 1',
    description: 'Normalize each source token over its D features.', inputShape: ['B', 'S', 'D'], outputShape: ['B', 'S', 'D'], operation: 'LayerNorm(D)', repetition: 'Nₑ',
  },
  {
    id: 'enc-qkv', kind: 'linear', stage: 'Encoder', eyebrow: 'ATTENTION PROJECTIONS', label: 'Self-attention Q/K/V',
    description: 'The same normalized source states produce queries, keys, and values.', inputShape: ['B', 'S', 'D'], outputShape: ['B', 'H', 'S', 'dₕ'],
    operation: 'XWq, XWk, XWv → reshape → transpose', ports: qkvOutputs, repetition: 'Nₑ',
    shapeDerivation: 'B × S × D → B × S × (H·dₕ) → B × H × S × dₕ, with H·dₕ = D.',
    parameters: { 'Wq/Wk/Wv': 'D × D', heads: 'H', 'head dim': 'dₕ = D/H' },
  },
  {
    id: 'enc-attn', kind: 'attention', stage: 'Encoder', eyebrow: 'BIDIRECTIONAL SELF-ATTENTION', label: 'Encoder attention',
    description: 'Every source query can attend to every non-padding source key.', inputShape: ['B', 'H', 'S', 'dₕ'], outputShape: ['B', 'S', 'D'],
    operation: 'softmax(QKᵀ / √dₕ) V → concat heads → Wo', ports: qkvInputs, repetition: 'Nₑ',
    formula: String.raw`A=\mathrm{softmax}(QK^T/\sqrt{d_h}),\quad A:[B,H,S,S]`,
    shapeDerivation: 'QKᵀ: (B,H,S,dₕ) × (B,H,dₕ,S) → B × H × S × S. A·V returns B × H × S × dₕ; concatenating H heads restores B × S × D.',
    parameters: { heads: 'H', 'head dim': 'dₕ', 'attention map': 'B × H × S × S' },
  },
  {
    id: 'enc-add1', kind: 'fusion', stage: 'Encoder', eyebrow: 'RESIDUAL ADD', label: 'Self-attention residual',
    description: 'Add the unchanged block input to the projected attention result.', inputShape: ['B', 'S', 'D'], outputShape: ['B', 'S', 'D'], operation: 'x + SelfAttention(LN(x))', repetition: 'Nₑ',
  },
  {
    id: 'enc-ln2', kind: 'normalization', stage: 'Encoder', eyebrow: 'PRE-NORM', label: 'Encoder LayerNorm 2',
    description: 'Normalize the residual stream before the position-wise FFN.', inputShape: ['B', 'S', 'D'], outputShape: ['B', 'S', 'D'], operation: 'LayerNorm(D)', repetition: 'Nₑ',
  },
  {
    id: 'enc-ffn', kind: 'linear', stage: 'Encoder', eyebrow: 'POSITION-WISE MLP', label: 'Encoder feed-forward',
    description: 'Expand and reduce each source position independently.', inputShape: ['B', 'S', 'D'], outputShape: ['B', 'S', 'D'],
    operation: 'Linear(D,D_ff) → activation → Linear(D_ff,D)', repetition: 'Nₑ',
    shapeDerivation: 'B × S × D → B × S × D_ff → B × S × D; sequence length S is unchanged.', parameters: { expansion: 'D → D_ff', reduction: 'D_ff → D' },
  },
  {
    id: 'enc-add2', kind: 'fusion', stage: 'Encoder', eyebrow: 'BLOCK OUTPUT', label: 'Encoder FFN residual',
    description: 'Complete one representative encoder block. Its output becomes the next layer input.', inputShape: ['B', 'S', 'D'], outputShape: ['B', 'S', 'D'], operation: 'y + FFN(LN(y))', repetition: 'Nₑ',
  },
]

const memoryNode: ArchitectureNode = { ...overviewNodes[3], id: 'enc-memory' }

const collapsedDecoder: ArchitectureNode = {
  ...overviewNodes[6], id: 'dec-block', label: 'Decoder block', eyebrow: 'COLLAPSED · EXPAND TO INSPECT',
  notes: ['The complete block is repeated N_d times; expanding shows one representative layer.'],
}

const expandedDecoder: ArchitectureNode[] = [
  {
    id: 'dec-ln1', kind: 'normalization', stage: 'Decoder', eyebrow: 'PRE-NORM', label: 'Decoder LayerNorm 1',
    description: 'Normalize the target stream before causal self-attention.', inputShape: ['B', 'T', 'D'], outputShape: ['B', 'T', 'D'], operation: 'LayerNorm(D)', repetition: 'N_d',
  },
  {
    id: 'dec-qkv', kind: 'linear', stage: 'Decoder', eyebrow: 'SELF-ATTENTION PROJECTIONS', label: 'Masked Q/K/V',
    description: 'Target states produce Q, K, and V for causal self-attention.', inputShape: ['B', 'T', 'D'], outputShape: ['B', 'H', 'T', 'dₕ'],
    operation: 'YWq, YWk, YWv → split heads', ports: qkvOutputs, repetition: 'N_d',
    shapeDerivation: 'B × T × D → B × H × T × dₕ, where H·dₕ = D.', parameters: { heads: 'H', 'head dim': 'dₕ = D/H' },
  },
  {
    id: 'dec-self-attn', kind: 'attention', stage: 'Decoder', eyebrow: 'CAUSAL SELF-ATTENTION', label: 'Masked self-attention',
    description: 'A triangular causal mask prevents target query t from reading keys after t.', inputShape: ['B', 'H', 'T', 'dₕ'], outputShape: ['B', 'T', 'D'],
    operation: 'softmax(QKᵀ/√dₕ + causal mask)V', ports: qkvInputs, repetition: 'N_d',
    formula: String.raw`A_{ij}=0\;\mathrm{for}\;j>i`, shapeDerivation: 'Attention scores have shape B × H × T × T. Concatenated head outputs restore B × T × D.',
    parameters: { heads: 'H', 'attention map': 'B × H × T × T', mask: 'causal + padding' },
  },
  {
    id: 'dec-add1', kind: 'fusion', stage: 'Decoder', eyebrow: 'RESIDUAL ADD', label: 'Masked-attention residual',
    description: 'Add the original target stream to masked self-attention output.', inputShape: ['B', 'T', 'D'], outputShape: ['B', 'T', 'D'], operation: 'y + MaskedAttention(LN(y))', repetition: 'N_d',
  },
  {
    id: 'cross-ln', kind: 'normalization', stage: 'Decoder', eyebrow: 'CROSS-ATTENTION PRE-NORM', label: 'Decoder LayerNorm 2',
    description: 'Normalize decoder states before querying encoder memory.', inputShape: ['B', 'T', 'D'], outputShape: ['B', 'T', 'D'], operation: 'LayerNorm(D)', repetition: 'N_d',
  },
  {
    id: 'cross-q', kind: 'linear', stage: 'Decoder', eyebrow: 'QUERY FROM DECODER', label: 'Cross-attention Q',
    description: 'Queries come from normalized decoder states.', inputShape: ['B', 'T', 'D'], outputShape: ['B', 'H', 'T', 'dₕ'], operation: 'Q = YWq → split heads',
    ports: [{ id: 'input', label: 'decoder', direction: 'input' }, { id: 'q', label: 'Q', direction: 'output' }], repetition: 'N_d',
  },
  {
    id: 'cross-kv', kind: 'linear', stage: 'Decoder', eyebrow: 'KEY/VALUE FROM ENCODER', label: 'Cross-attention K/V',
    description: 'Keys and values come from encoder memory, so their sequence axis is S rather than T.', inputShape: ['B', 'S', 'D'], outputShape: ['B', 'H', 'S', 'dₕ'], operation: 'K = EWk, V = EWv → split heads',
    ports: [{ id: 'input', label: 'memory', direction: 'input' }, { id: 'k', label: 'K', direction: 'output' }, { id: 'v', label: 'V', direction: 'output' }], repetition: 'N_d',
  },
  {
    id: 'cross-attn', kind: 'attention', stage: 'Decoder', eyebrow: 'ENCODER–DECODER ATTENTION', label: 'Cross-attention',
    description: 'Each target query distributes attention over all valid source keys.', inputShape: ['B', 'H', 'T/S', 'dₕ'], outputShape: ['B', 'T', 'D'],
    operation: 'softmax(Qdecoder Kencoderᵀ / √dₕ) Vencoder', ports: qkvInputs, repetition: 'N_d',
    formula: 'QK^T:[B,H,T,S]', shapeDerivation: 'Q is B × H × T × dₕ while K and V are B × H × S × dₕ. Scores are B × H × T × S; multiplying V and merging heads gives B × T × D.',
    parameters: { heads: 'H', 'query length': 'T', 'key/value length': 'S', 'attention map': 'B × H × T × S' },
  },
  {
    id: 'dec-add2', kind: 'fusion', stage: 'Decoder', eyebrow: 'RESIDUAL ADD', label: 'Cross-attention residual',
    description: 'Add cross-attention context to the masked self-attention stream.', inputShape: ['B', 'T', 'D'], outputShape: ['B', 'T', 'D'], operation: 'z + CrossAttention(LN(z), memory)', repetition: 'N_d',
  },
  {
    id: 'dec-ln3', kind: 'normalization', stage: 'Decoder', eyebrow: 'FFN PRE-NORM', label: 'Decoder LayerNorm 3',
    description: 'Normalize target states before the decoder FFN.', inputShape: ['B', 'T', 'D'], outputShape: ['B', 'T', 'D'], operation: 'LayerNorm(D)', repetition: 'N_d',
  },
  {
    id: 'dec-ffn', kind: 'linear', stage: 'Decoder', eyebrow: 'POSITION-WISE MLP', label: 'Decoder feed-forward',
    description: 'Expand and reduce every target position independently.', inputShape: ['B', 'T', 'D'], outputShape: ['B', 'T', 'D'], operation: 'Linear(D,D_ff) → activation → Linear(D_ff,D)', repetition: 'N_d',
    shapeDerivation: 'B × T × D → B × T × D_ff → B × T × D.', parameters: { expansion: 'D → D_ff', reduction: 'D_ff → D' },
  },
  {
    id: 'dec-add3', kind: 'fusion', stage: 'Decoder', eyebrow: 'BLOCK OUTPUT', label: 'Decoder FFN residual',
    description: 'Complete one representative decoder block.', inputShape: ['B', 'T', 'D'], outputShape: ['B', 'T', 'D'], operation: 'u + FFN(LN(u))', repetition: 'N_d',
  },
]

const outputNode: ArchitectureNode = { ...overviewNodes[7], id: 'logits' }

function detailedEdges(expandEncoder: boolean, expandDecoder: boolean): ArchitectureEdge[] {
  const encEntry = expandEncoder ? 'enc-ln1' : 'enc-block'
  const encExit = expandEncoder ? 'enc-add2' : 'enc-block'
  const decExit = expandDecoder ? 'dec-add3' : 'dec-block'
  const edges: ArchitectureEdge[] = [
    { id: 'src-srcembed', source: 'source', target: 'src-embed', kind: 'data' },
    { id: 'srcembed-encoder', source: 'src-embed', target: encEntry, kind: 'data' },
    { id: 'encoder-memory', source: encExit, target: 'enc-memory', kind: 'data' },
    { id: 'target-tgtembed', source: 'target', target: 'tgt-embed', kind: 'data' },
    { id: 'decoder-logits', source: decExit, target: 'logits', kind: 'data' },
  ]

  if (expandEncoder) edges.push(
    { id: 'enc-ln-qkv', source: 'enc-ln1', target: 'enc-qkv', kind: 'data' },
    { id: 'enc-q', source: 'enc-qkv', sourcePort: 'q', target: 'enc-attn', targetPort: 'q', kind: 'data', label: 'Q · B×H×S×dₕ' },
    { id: 'enc-k', source: 'enc-qkv', sourcePort: 'k', target: 'enc-attn', targetPort: 'k', kind: 'data', label: 'K' },
    { id: 'enc-v', source: 'enc-qkv', sourcePort: 'v', target: 'enc-attn', targetPort: 'v', kind: 'data', label: 'V' },
    { id: 'enc-attn-add', source: 'enc-attn', target: 'enc-add1', kind: 'data' },
    { id: 'enc-residual1', source: 'src-embed', target: 'enc-add1', kind: 'residual', label: 'residual' },
    { id: 'enc-add-ln2', source: 'enc-add1', target: 'enc-ln2', kind: 'data' },
    { id: 'enc-ln2-ffn', source: 'enc-ln2', target: 'enc-ffn', kind: 'data' },
    { id: 'enc-ffn-add2', source: 'enc-ffn', target: 'enc-add2', kind: 'data' },
    { id: 'enc-residual2', source: 'enc-add1', target: 'enc-add2', kind: 'residual', label: 'residual' },
  )

  if (expandDecoder) edges.push(
    { id: 'tgt-dec-ln', source: 'tgt-embed', target: 'dec-ln1', kind: 'data' },
    { id: 'dec-ln-qkv', source: 'dec-ln1', target: 'dec-qkv', kind: 'data' },
    { id: 'dec-q', source: 'dec-qkv', sourcePort: 'q', target: 'dec-self-attn', targetPort: 'q', kind: 'data', label: 'Q' },
    { id: 'dec-k', source: 'dec-qkv', sourcePort: 'k', target: 'dec-self-attn', targetPort: 'k', kind: 'data', label: 'K' },
    { id: 'dec-v', source: 'dec-qkv', sourcePort: 'v', target: 'dec-self-attn', targetPort: 'v', kind: 'data', label: 'V' },
    { id: 'dec-attn-add1', source: 'dec-self-attn', target: 'dec-add1', kind: 'data' },
    { id: 'dec-residual1', source: 'tgt-embed', target: 'dec-add1', kind: 'residual', label: 'residual' },
    { id: 'dec-add-crossln', source: 'dec-add1', target: 'cross-ln', kind: 'data' },
    { id: 'crossln-q', source: 'cross-ln', target: 'cross-q', kind: 'data' },
    { id: 'memory-kv', source: 'enc-memory', target: 'cross-kv', kind: 'data', label: 'B×S×D' },
    { id: 'cross-q-attn', source: 'cross-q', sourcePort: 'q', target: 'cross-attn', targetPort: 'q', kind: 'data', label: 'Q · B×H×T×dₕ' },
    { id: 'cross-k-attn', source: 'cross-kv', sourcePort: 'k', target: 'cross-attn', targetPort: 'k', kind: 'data', label: 'K · B×H×S×dₕ' },
    { id: 'cross-v-attn', source: 'cross-kv', sourcePort: 'v', target: 'cross-attn', targetPort: 'v', kind: 'data', label: 'V' },
    { id: 'cross-attn-add2', source: 'cross-attn', target: 'dec-add2', kind: 'data' },
    { id: 'dec-residual2', source: 'dec-add1', target: 'dec-add2', kind: 'residual', label: 'residual' },
    { id: 'dec-add2-ln3', source: 'dec-add2', target: 'dec-ln3', kind: 'data' },
    { id: 'dec-ln3-ffn', source: 'dec-ln3', target: 'dec-ffn', kind: 'data' },
    { id: 'dec-ffn-add3', source: 'dec-ffn', target: 'dec-add3', kind: 'data' },
    { id: 'dec-residual3', source: 'dec-add2', target: 'dec-add3', kind: 'residual', label: 'residual' },
  )
  else edges.push(
    { id: 'tgt-decoder', source: 'tgt-embed', target: 'dec-block', targetPort: 'target', kind: 'data' },
    { id: 'memory-decoder', source: 'enc-memory', target: 'dec-block', targetPort: 'context', kind: 'data', label: 'K, V context' },
  )

  return edges
}

export function createTransformerModel(projection: ArchitectureProjection, options: ArchitectureOptions = {}): ArchitectureModel {
  if (projection === 'overview') return { ...common, projection, nodes: overviewNodes, edges: overviewEdges }
  const expandEncoder = options.expandEncoder ?? false
  const expandDecoder = options.expandDecoder ?? false
  return {
    ...common,
    projection,
    nodes: [
      ...sourceInputNodes,
      ...(expandEncoder ? expandedEncoder : [collapsedEncoder]),
      memoryNode,
      ...targetInputNodes,
      ...(expandDecoder ? expandedDecoder : [collapsedDecoder]),
      outputNode,
    ],
    edges: detailedEdges(expandEncoder, expandDecoder),
  }
}

export const transformerArchitecture: ArchitectureDefinition = {
  id: 'transformer-encoder-decoder',
  badge: 'SEQ2SEQ',
  context: 'MODEL ATLAS / TRANSFORMERS',
  metrics: [
    { value: 'S → T', label: 'SEQUENCE' },
    { value: 'D', label: 'D_MODEL' },
    { value: 'H', label: 'HEADS' },
    { value: 'D/H', label: 'HEAD DIM' },
    { value: 'Nₑ/N_d', label: 'LAYERS' },
  ],
  stages: [
    { label: 'Inputs', color: '#4fc0a2' },
    { label: 'Encoder', color: '#e5a54a' },
    { label: 'Decoder', color: '#a879e6' },
    { label: 'Output', color: '#ed6a8c' },
  ],
  toggles: [
    { id: 'expandEncoder', label: 'Expand encoder', activeLabel: 'Collapse encoder', projection: 'detail' },
    { id: 'expandDecoder', label: 'Expand decoder', activeLabel: 'Collapse decoder', projection: 'detail' },
  ],
  sources: ['Attention Is All You Need (Vaswani et al., 2017)', 'standard pre-norm variant'],
  footerNote: 'Symbolic assumptions: B=batch · S=source length · T=target length · D=model width · H=heads · dₕ=D/H · Vtgt=vocabulary',
  createModel: createTransformerModel,
}
