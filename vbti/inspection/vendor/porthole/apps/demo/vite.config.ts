import { fileURLToPath } from 'node:url';

import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

const fromHere = (path: string) => fileURLToPath(new URL(path, import.meta.url));

export default defineConfig({
  plugins: [react()],
  // Relative base so the built bundle works when Python serves it from any path.
  base: './',
  resolve: {
    // Point at source, not dist: in-repo edits to the framework or a registry
    // panel show up through HMR without a build step. External consumers get
    // the same files a different way — the package for the framework, a copy
    // in their repo for the panels.
    //
    // Array form with an anchored regex, because the bare specifier and its
    // `/theme.css` subpath must resolve to different files; a plain string
    // alias would rewrite the subpath into `.../index.ts/theme.css`.
    alias: [
      {
        find: '@porthole/framework/theme.css',
        replacement: fromHere('../../packages/framework/src/theme/theme.css'),
      },
      {
        find: /^@porthole\/framework$/,
        replacement: fromHere('../../packages/framework/src/index.ts'),
      },
      {
        find: '@porthole/registry',
        replacement: fromHere('../../packages/registry/src'),
      },
    ],
  },
  server: {
    port: 5173,
    // Reachable from another machine, which is the whole point of supporting
    // the browser as a target.
    host: true,
  },
});
