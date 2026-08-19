import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// Unlike porthole's own demo, this app consumes `@porthole/framework` as a real
// package (a `file:` dependency resolving to its built `dist/`). That is the
// path an external consumer takes, so building here also checks that the
// framework's published surface is actually sufficient — an alias to source
// would hide a missing export.
export default defineConfig({
  plugins: [react()],
  // Relative base: Python serves the bundle from whatever origin it likes.
  base: './',
  resolve: {
    /**
     * Load exactly one copy of anything stateful.
     *
     * A `file:` dependency is a symlink, so Node resolves the framework's own
     * imports from *porthole's* node_modules — a second React, a second
     * dockview. Two Reacts means the hook dispatcher is null in whichever tree
     * did not create the root, and the whole app dies on the first `useMemo`
     * with an error that names neither React nor the symlink.
     *
     * `dedupe` forces every one of these to resolve to this app's copy. Any
     * package with module-level state that crosses the framework boundary
     * belongs on this list.
     */
    dedupe: [
      'react',
      'react-dom',
      'dockview-react',
      'dockview-core',
      'three',
      '@react-three/fiber',
      '@react-three/drei',
    ],
  },
  build: { outDir: 'dist', emptyOutDir: true },
  server: { port: 5174, host: true },
});
