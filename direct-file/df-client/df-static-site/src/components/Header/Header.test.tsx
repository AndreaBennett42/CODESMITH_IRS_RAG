import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { I18nextProvider } from 'react-i18next';
import { getI18nForTest } from 'df-i18n';

import Header from './Header.js';

describe(`Header component`, () => {
  const i18n = getI18nForTest();
  test(`renders without errors`, () => {
    render(
      <I18nextProvider i18n={i18n}>
        <BrowserRouter>
          <Header />
        </BrowserRouter>
      </I18nextProvider>
    );
    const logoLink = screen.getAllByRole(`link`)[0];
    expect(logoLink).toHaveAttribute(`rel`, `home`);
  });
});
