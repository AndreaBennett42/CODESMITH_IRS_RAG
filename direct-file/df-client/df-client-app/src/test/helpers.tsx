import { ReactNode } from 'react';
import { HelmetProvider } from 'react-helmet-async';
import { BrowserRouter } from 'react-router-dom';

/**
 * Assist testing components which require a router or helmet provider.
 */
export const wrapComponent = (component: ReactNode) => (
  <HelmetProvider>
    <BrowserRouter>{component}</BrowserRouter>
  </HelmetProvider>
);
