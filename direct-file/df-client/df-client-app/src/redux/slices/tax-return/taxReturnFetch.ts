import { read } from '../../../hooks/useApiHook.js';
import { TaxReturn } from '../../../types/core.js';

/**
 * Performs the network fetch for tax returns.
 *
 * This encapsulates our fetch promise in a way that is easy to use `spy` on in tests
 * previously, we just used `spy` on a fetch which has the downside that we can't
 * then trigger any logging on this fetch.
 */
export function taxReturnFetch(): Promise<TaxReturn[]> {
  return read<TaxReturn[]>(`${import.meta.env.VITE_BACKEND_URL}v1/taxreturns`, {});
}
