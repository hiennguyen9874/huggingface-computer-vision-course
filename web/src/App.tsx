import { ArchitectureExplorer } from './modules/architecture-explorer'
import { vitArchitecture } from './models/vit'

export default function App() {
  return <ArchitectureExplorer definition={vitArchitecture} />
}
