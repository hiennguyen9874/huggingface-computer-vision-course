# Architecture models

Each model lives in its own directory and exports an `ArchitectureDefinition`.
The definition owns model-specific metadata, controls, and a `createModel` function
that projects model data into overview or detailed graph IR.

To add an architecture:

1. Create `models/<model>/` with its model data and an `index.ts` export.
2. Define nodes and edges using the types exported by `modules/architecture-explorer`.
3. Export an `ArchitectureDefinition` with metrics, stages, sources, optional toggles,
   and a projection factory.
4. Pass that definition to `<ArchitectureExplorer definition={...} />` in `App.tsx`,
   or add it to a future model picker.

Rendering, path highlighting, inspection, and ELK layout belong in
`modules/architecture-explorer`; model dimensions and formulas belong here.
