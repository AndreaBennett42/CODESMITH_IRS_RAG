import { render } from '@testing-library/react';
import { store } from '../redux/store.js';
import { TaxReturn } from '../types/core.js';
import { BrowserRouter } from 'react-router-dom';
import { v4 as uuidv4 } from 'uuid';
import { SystemAlertContextProvider } from './SystemAlertContext/SystemAlertContext.js';
import { TaxReturnsContextProvider } from './TaxReturnsContext.js';
import { Provider } from 'react-redux';
import { taxReturnFetch } from '../redux/slices/tax-return/taxReturnFetch.js';
import { Mock, describe, expect, it, vi } from 'vitest';

vi.mock(`../redux/slices/tax-return/taxReturnFetch.js`, () => ({
  taxReturnFetch: vi.fn(), // Mock the function
}));

const now = new Date();
const fakeTaxReturns: TaxReturn[] = [
  {
    taxYear: 2023,
    id: `foo`,
    createdAt: new Date().toISOString(),
    taxReturnSubmissions: [
      {
        id: uuidv4(),
        receiptId: `12345620230215000001`,
        submitUserId: `Alex`,
        createdAt: now.setDate(now.getDate() - 1).toString(),
        submissionReceivedAt: now.setDate(now.getDate() - 1).toString(),
      },
    ],
    facts: {},
    isEditable: false,
    surveyOptIn: null,
  },
];

describe(`Tax Returns context `, () => {
  it(`will fetch tax returns if they have not yet been loaded`, () => {
    (taxReturnFetch as Mock).mockReturnValue(Promise.resolve(fakeTaxReturns));
    render(
      <BrowserRouter>
        <Provider store={store}>
          <SystemAlertContextProvider>
            <TaxReturnsContextProvider>
              <div />
            </TaxReturnsContextProvider>
          </SystemAlertContextProvider>
        </Provider>
      </BrowserRouter>
    );

    expect(taxReturnFetch).toHaveBeenCalledTimes(1);
  });

  it(`won't fetch tax returns if they have been loaded`, () => {
    (taxReturnFetch as Mock).mockReturnValue(Promise.resolve(fakeTaxReturns));

    const testComponents = (
      <BrowserRouter>
        <Provider store={store}>
          <SystemAlertContextProvider>
            <TaxReturnsContextProvider>
              <div />
            </TaxReturnsContextProvider>
          </SystemAlertContextProvider>
        </Provider>
      </BrowserRouter>
    );

    const { rerender } = render(testComponents);

    // Fetches tax returns on init
    expect(taxReturnFetch).toHaveBeenCalledTimes(1);

    rerender(testComponents);

    // Subsequent renders do not increase the number of times tax returns have been fetched
    expect(taxReturnFetch).toHaveBeenCalledTimes(1);
  });
});
