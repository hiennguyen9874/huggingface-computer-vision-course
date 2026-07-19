import { useState } from 'react'
import { ArchitectureExplorer } from './modules/architecture-explorer'
import { transformerArchitecture } from './models/transformer'
import { vitArchitecture } from './models/vit'

const architectures = [vitArchitecture, transformerArchitecture]

export default function App() {
  const [architectureId, setArchitectureId] = useState(transformerArchitecture.id)
  const definition = architectures.find((architecture) => architecture.id === architectureId) ?? transformerArchitecture

  const navigation = (
    <nav className="model-switch" aria-label="Architecture model">
      <span>MODEL</span>
      {architectures.map((architecture) => (
        <button
          key={architecture.id}
          className={architecture.id === architectureId ? 'active' : ''}
          onClick={() => setArchitectureId(architecture.id)}
        >
          {architecture.badge}
        </button>
      ))}
    </nav>
  )

  return <ArchitectureExplorer key={definition.id} definition={definition} navigation={navigation} />
}
