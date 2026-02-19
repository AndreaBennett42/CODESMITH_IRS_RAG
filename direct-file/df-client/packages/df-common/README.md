# Common Components

This library has a set of shared React components that should look and feel the same across both `df-client-app` and `df-static-site`.

These components serve as wrappers that contain the styles and allow for translations and child elements to be passed to them.

## How to use

Import the component as you would any other React library.

```tsx
import { CommonHeader } from '@irs/df-common';
```

## How to add a component

Create a folder named after the component inside `src/components`. Create a `.tsx` file named after the component and a `index.tsx` file to export it.

Inside `src/components/index.ts`, make sure to add a line to export the component here as well:

```ts
export { default as CommonPilotBanner } from './components/CommonPilotBanner.js';
```

### Naming

Components in this library are all namespaced with the "Common" prefix.
