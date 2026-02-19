import { ReactNode } from 'react';
import { HelmetProvider } from 'react-helmet-async';
import { MemoryRouter } from 'react-router-dom';

/**
 * Assist testing components which require a router or helmet provider.
 */
export const wrapComponent = (component: ReactNode, options: { route: string } = { route: `/` }) => (
  <HelmetProvider>
    <MemoryRouter initialEntries={[options.route]}>{component}</MemoryRouter>
  </HelmetProvider>
);
