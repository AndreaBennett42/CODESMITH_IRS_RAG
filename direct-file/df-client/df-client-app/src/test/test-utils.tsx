/* eslint-disable import/export */
import { ReactElement, ReactNode } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { HelmetProvider } from 'react-helmet-async';
import { BrowserRouter } from 'react-router-dom';
import { I18nextProvider } from 'react-i18next';
import { getI18nForTest } from 'df-i18n';

const i18n = getI18nForTest();

const AllTheProviders = ({ children }: { children: ReactNode }) => {
  return (
    <I18nextProvider i18n={i18n}>
      <HelmetProvider>
        <BrowserRouter>{children}</BrowserRouter>
      </HelmetProvider>
    </I18nextProvider>
  );
};

const TranslationProvider = ({ children }: { children: ReactNode }) => {
  return <I18nextProvider i18n={i18n}>{children}</I18nextProvider>;
};

const renderWithAllProviders = (ui: ReactElement, options?: Omit<RenderOptions, 'wrapper'>) =>
  render(ui, { wrapper: AllTheProviders, ...options });

const renderWithTranslationProvider = (ui: ReactElement, options?: Omit<RenderOptions, 'wrapper'>) =>
  render(ui, { wrapper: TranslationProvider, ...options });

export * from '@testing-library/react';
export { renderWithAllProviders, renderWithTranslationProvider, i18n };
