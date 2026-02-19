import { defineConfig } from 'vitest/config';
import { configOptions } from './vite.config.js';
import { viteSingleFile } from 'vite-plugin-singlefile';
import { PluginOption } from 'vite';
import path from 'path';
import browserslistToEsbuild from 'browserslist-to-esbuild';

/**
 * Vite Configuration for Single-File Build
 *
 * PROBLEM:
 * When building a single HTML file using vite-plugin-singlefile, we have been encountering intermittent issues
 * with Vite's runtime variables. These variables are normally injected by Vite's dev server during development
 * or handled by its production build process. However, when creating a standalone HTML file that
 * needs to run without a server, these variables seem to be missing and cause runtime errors.
 *
 * SOLUTION:
 * We inject all necessary Vite runtime variables directly into the window object using rollupOptions.output.intro.
 * This approach:
 * 1. Avoids the need for Vite's dev server
 * 2. Makes the build truly standalone
 * 3. Prevents "undefined variable" errors that would normally occur
 *
 * WHY THIS WORKS:
 * - The 'intro' option in rollup adds code at the beginning of the bundle
 * - By adding variables to the window object, we ensure global accessibility
 * - Setting these variables manually replaces Vite's runtime injection
 * - Using window.* instead of direct declarations prevents issues with strict mode
 *
 * VARIABLES EXPLAINED:
 * - __VITE_PRELOAD__: Required for Vite's module preloading
 * - __DEFINES__: Used for Vite's define plugin
 * - __HMR_*: Hot Module Replacement related variables (needed even in prod)
 * - __MODE__, __DEV__, __PROD__: Environment indicators
 *
 * Note: While many of these variables are primarily for development (like HMR),
 * they're still referenced in the production build and must be defined to prevent errors.
 */
export default defineConfig({
  ...configOptions,
  ...{
    plugins: [...(configOptions.plugins as PluginOption[]), viteSingleFile()],
    build: {
      outDir: 'dist-all-screens',
      target: browserslistToEsbuild(['>0.2%', 'not dead', 'not op_mini all']),
      modulePreload: false,
      rollupOptions: {
        input: {
          main: path.resolve(__dirname, 'all-screens/index.html'),
        },
        output: {
          intro: `
            window.__VITE_PRELOAD__ = () => {};
            window.__DEFINES__ = {};
            window.__HMR_CONFIG_NAME__ = {};
            window.__BASE__ = "/";
            window.__SERVER_HOST__ = "localhost";
            window.__HMR_PROTOCOL__ = "ws";
            window.__HMR_PORT__ = "3000";
            window.__HMR_HOSTNAME__ = "localhost";
            window.__HMR_BASE__ = "/";
            window.__HMR_DIRECT_TARGET__ = "";
            window.__HMR_TIMEOUT__ = 30000;
            window.__HMR_ENABLE_OVERLAY__ = false;
            window.__MODE__ = "production";
            window.__DEV__ = false;
            window.__PROD__ = true;
          `,
        },
      },
    },
  },
});
