import { useState, Dispatch, SetStateAction, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { Fetch, FetchArgs, FetchResponse, UseFetchPdfResponse, UseSaveResponse } from '../types/core.js';
import {
  formatAndAppendHeaders,
  formatQueryParams,
  ReadError,
  handledErrors,
  PII_SERVICE_ERROR,
} from '../misc/apiHelpers.js';
import { SessionClock } from '../auth/session.js';
import { useEffectOnce } from './useEffectOnce.js';

export type ReadOptions<T> = {
  headers?: { [key: string]: string };
  queryParams?: string;
  method?: `GET`;
  hooks?: {
    setStatus?: Dispatch<SetStateAction<number>>;
    setData?: Dispatch<SetStateAction<T>>;
    setLoading?: Dispatch<SetStateAction<boolean>>;
    setSuccess?: Dispatch<SetStateAction<boolean>>;
  };
  initialState?: {
    loading?: boolean;
  };
};

export type WriteOptions<T> = Omit<ReadOptions<T>, 'method'> & {
  method?: `POST`;
  body?: object;
};

const baseRouterPath = import.meta.env.VITE_PUBLIC_PATH || ``;

/** Given an error constant, go to the associated error page. */
const navigateToErrorPage = (error: string) => {
  const errorPage = baseRouterPath + handledErrors[error];
  // This is not the prettiest but we are not within the Router here
  if (window.location.pathname !== errorPage) {
    window.location.pathname = errorPage;
  }
};

/**
 * Do things before or after `fetch` is run, globally.
 *
 * Downstream users of `fetch` should be able to rely on substantially the same
 * behavior as native fetch. . . be careful when making modifications here.
 *
 * Used as `window.fetch = interceptor(fetch); in index.tsx`.
 */
export const interceptor = (_fetch: Fetch) => {
  const clock = new SessionClock();
  clock.start();

  /**
   * Helper function. Intercept and examine the result of a successful API call.
   * Note that "successful" means no network errors, not "no errors at all".
   */
  const afterFetch = async (response: Response, args: FetchArgs) => {
    const contentType = response.headers.get(`content-type`) ?? ``;
    const url = args.length && typeof args[0] === `string` ? args[0] : ``;
    const isBackend = url.startsWith(`${import.meta.env.VITE_BACKEND_URL}`);

    if (isBackend) {
      if (response.status === 200 && contentType.indexOf(`text/html`) !== -1) {
        // an HTML page probably means that the Site Minder / SADI session has ended
        clock.alertUser();
      } else {
        // any other response means the session is probably still alive
        clock.reset();
      }
    }

    if (contentType.indexOf(`application/json`) !== -1) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let data: any; // response from json() is Promise<any>
      try {
        // read the JSON body -- it may contain an error message
        data = await response.json();
      } catch (error) {
        // some endpoints are `application/json` but have no body
        data = undefined;
      }
      const isObject = typeof data === `object`;
      const hasMessage = isObject && Object.prototype.hasOwnProperty.call(data, `message`);
      const message = hasMessage ? data[`message`] : ``;

      // 403 will be generated for reasons detailed in `handledErrors`
      if (response.status === 403 && message in handledErrors) {
        navigateToErrorPage(message);
        throw new ReadError(message, response.status);
      }
      // 401 will be generated for SADI being unresponsive
      if (response.status === 401 && message === PII_SERVICE_ERROR) {
        // these errors are caught by LoadingVerify.tsx
        throw new ReadError(PII_SERVICE_ERROR, response.status);
      }
      // calling json() used up the promise, but callers of fetch expect one, so fake it
      response.json = () => Promise.resolve(data);
    }
    return response;
  };

  // return a modified version of fetch
  return (...args: FetchArgs) => {
    return _fetch.apply(this, args).then((response) => afterFetch(response, args));
  };
};

interface Translateable {
  apiErrorKey: string;
  body?: object;
}

export const isTranslateable = (maybeTranslateable: unknown): maybeTranslateable is Translateable => {
  return typeof maybeTranslateable === `object` && maybeTranslateable !== null && `apiErrorKey` in maybeTranslateable;
};

// TODO: Some functions calling this are using the promise, some try/catch blocks
//       Need to standardize on one pattern.
/**
 * Make API calls to the backend. Returns JSON data. This is a wrapper around fetch.
 *
 * Use as:
 * ```
 *  try {
 *    await read<TypeOfResponse>(`${import.meta.env.VITE_BACKEND_URL}v1/path/to/thing`)
 *  } catch (error) {
 *    ... handle failure ...
 *  }
 * ```
 */
export async function read<T>(url: string, { headers, queryParams, hooks = {} }: ReadOptions<T> = {}): Promise<T> {
  let response: Response, json;
  const formattedUrl = formatQueryParams(queryParams ?? ``, url);
  const formattedHeaders = formatAndAppendHeaders(headers ?? {});
  hooks.setLoading?.(true);

  try {
    // nosemgrep: nodejs_scan.javascript-ssrf-rule-node_ssrf
    response = await fetch(formattedUrl, {
      method: `GET`,
      headers: formattedHeaders,
    });
    hooks.setStatus?.(response.status);
  } catch (error) {
    hooks.setLoading?.(false);
    hooks.setSuccess?.(false);
    throw new ReadError(`Network error`, null, error);
  }

  if (!response.ok) {
    hooks.setLoading?.(false);
    hooks.setSuccess?.(false);
    throw new ReadError(`Response was not successful`, response.status);
  }

  try {
    json = await response.json();
  } catch (error) {
    hooks.setSuccess?.(false);
    throw new ReadError(`Failed to parse JSON response`, response.status, error);
  } finally {
    hooks.setLoading?.(false);
  }

  hooks.setSuccess?.(true);
  hooks.setData?.(json as T);
  return json as T;
}

const EMPTY_OBJECT = {};

/**
 * A wrapper around `read`. It swallows errors and presents a tidy, stateful interface.
 *
 * Use as:
 * ```
 *  const { loading, success, data } = useRead<TypeOfResponse>(`${import.meta.env.VITE_BACKEND_URL}v1/path/to/thing`);
 * ```
 */
export function useRead<T>(url: string, options: ReadOptions<T> = EMPTY_OBJECT): FetchResponse<T> {
  const readOptions: ReadOptions<T> = {
    ...options,
    initialState: {
      loading: true,
    },
  };
  const { data, loading, success, callRead } = useReadManualCall(url, readOptions);
  useEffectOnce(() => {
    callRead();
  }, []);

  return { data: data as T, loading, success, callRead };
}

/** A wrapper around `read`. Operated similar to `useRead`, but read is not automatically called on mount */
export function useReadManualCall<T>(url: string, options: ReadOptions<T> = EMPTY_OBJECT): FetchResponse<T> {
  const [data, setData] = useState<T>();
  const [loading, setLoading] = useState<boolean>(options.initialState?.loading || false);
  const [success, setSuccess] = useState<boolean>(false);

  const run = useCallback(async () => {
    try {
      await read(url, { ...options, hooks: { setData, setLoading, setSuccess } });
    } catch (error) {
      return;
    }
  }, [options, url]);

  const callRead = useCallback(async () => {
    void run();
  }, [run]);

  return { data: data as T, loading, success, callRead };
}

/**
 * Make API calls to the backend. This is a wrapper around fetch.
 *
 * Returns JSON data if the response contains any. Returns `undefined` otherwise.
 *
 * Use as:
 * ```
 *  try {
 *    await save<TypeOfResponse>(
 *      `${import.meta.env.VITE_BACKEND_URL}v1/path/to/thing`,
 *      { body: ...data... }
 *    )
 *  } catch (error) {
 *    ... handle failure ...
 *  }
 * ```
 */
export async function save<T>(url: string, { method, headers, body, hooks = {} }: WriteOptions<T> = {}): Promise<T> {
  let response: Response, json;
  const formattedHeaders = formatAndAppendHeaders(headers ?? {});
  hooks.setLoading?.(true);

  try {
    // nosemgrep: nodejs_scan.javascript-ssrf-rule-node_ssrf
    response = await fetch(url, {
      method: method ?? `POST`,
      headers: formattedHeaders,
      body: JSON.stringify(body),
    });
    hooks.setStatus?.(response.status);
  } catch (error) {
    hooks.setLoading?.(false);
    hooks.setSuccess?.(false);
    throw new ReadError(`Network error`, null, error);
  }

  try {
    const isJSON = (response.headers.get(`content-type`) ?? ``).indexOf(`application/json`) !== -1;
    json = isJSON ? await response.json() : undefined;
  } catch {
    json = undefined;
  } finally {
    hooks.setLoading?.(false);
  }

  if (!response.ok) {
    hooks.setSuccess?.(false);
    if (json) {
      return Promise.reject(json);
    } else {
      throw new ReadError(`Response was not successful`, response.status);
    }
  }

  hooks.setSuccess?.(true);
  hooks.setData?.(json as T);
  return json as T;
}

/**
 * A wrapper around `save`. It swallows errors and presents a tidy, stateful interface.
 *
 * Use as:
 * ```
 *  const { loading, success, data, save } = useSave<TypeOfResponse>(
 *   `${import.meta.env.VITE_BACKEND_URL}v1/path/to/thing`,
 *   { body: ...data... }
 *  );
 *  save();
 * ```
 */
export function useSave<T>(url: string, options: WriteOptions<T> = {}): UseSaveResponse<T> {
  const [data, setData] = useState<T>();
  const [loading, setLoading] = useState<boolean>(false);
  const [success, setSuccess] = useState<boolean>(false);

  const callSave = () => {
    (async () => {
      try {
        await save(url, { ...options, hooks: { setData, setLoading, setSuccess } });
      } catch (error) {
        return;
      }
    })();
  };

  return { data: data as T, loading, success, save: callSave };
}

/**
 * Opens XML in a new window. This is a wrapper around fetch.
 *
 * Use as:
 * ```
 *  try {
 *    await fetchXML<TypeOfResponse>(...url...)
 *  } catch (error) {
 *    ... handle failure ...
 *  }
 * ```
 */

export async function fetchXml(url: string, { hooks = {} }: ReadOptions<null> = {}) {
  let response: Response;
  hooks.setLoading?.(true);

  try {
    // nosemgrep: nodejs_scan.javascript-ssrf-rule-node_ssrf
    response = await fetch(url, {
      method: `GET`,
      headers: formatAndAppendHeaders({ [`Content-Type`]: `text/xml` }),
    });
    hooks.setStatus?.(response.status);
  } catch (error) {
    hooks.setLoading?.(false);
    hooks.setSuccess?.(false);
    throw new ReadError(`Network error`, null, error);
  }

  if (!response.ok) {
    hooks.setLoading?.(false);
    hooks.setSuccess?.(false);
    throw new ReadError(`Response was not successful`, response.status);
  }

  try {
    const isXml = (response.headers.get(`content-type`) ?? ``).indexOf(`text/xml`) !== -1;
    if (isXml) {
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      window.open(url);
      hooks.setSuccess?.(true);
    } else {
      hooks.setSuccess?.(false);
      throw new ReadError(`Expected text/xml`, response.status);
    }
  } catch (error) {
    hooks.setSuccess?.(false);
    throw new ReadError(`Could not open XML`, response.status, error);
  } finally {
    hooks.setLoading?.(false);
  }
}

/**
 * Opens PDFs in a new window. This is a wrapper around fetch.
 *
 * Use as:
 * ```
 *  try {
 *    await fetchPdf<TypeOfResponse>(...url...)
 *  } catch (error) {
 *    ... handle failure ...
 *  }
 * ```
 */
export async function fetchPdf(url: string, { hooks = {} }: ReadOptions<null> = {}) {
  let response: Response;
  hooks.setLoading?.(true);

  try {
    // nosemgrep: nodejs_scan.javascript-ssrf-rule-node_ssrf
    response = await fetch(url, {
      method: `POST`,
      headers: formatAndAppendHeaders({ [`Content-Type`]: `application/pdf` }),
    });
    hooks.setStatus?.(response.status);
  } catch (error) {
    hooks.setLoading?.(false);
    hooks.setSuccess?.(false);
    throw new ReadError(`Network error`, null, error);
  }

  if (!response.ok) {
    hooks.setLoading?.(false);
    hooks.setSuccess?.(false);
    throw new ReadError(`Response was not successful`, response.status);
  }

  try {
    const isPDF = (response.headers.get(`content-type`) ?? ``).indexOf(`application/pdf`) !== -1;
    if (isPDF) {
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      window.open(url);
      hooks.setSuccess?.(true);
    } else {
      hooks.setSuccess?.(false);
      throw new ReadError(`Expected application/pdf`, response.status);
    }
  } catch (error) {
    hooks.setSuccess?.(false);
    throw new ReadError(`Could not open PDF`, response.status, error);
  } finally {
    hooks.setLoading?.(false);
  }
}

/**
 * A wrapper around `useFetchPdf`. It swallows errors and presents a tidy, stateful interface.
 *
 * Use as:
 * ```
 *  const { loading, success, callFetchPdf } = useFetchPdf(taxId);
 *  fetchPdf(taxId);
 * ```
 */
export const useFetchPdf = (): UseFetchPdfResponse => {
  const { i18n } = useTranslation();
  const [loading, setLoading] = useState<boolean>(false);
  const [success, setSuccess] = useState<boolean>(false);

  const callFetchPdf = (taxId: string) => {
    (async () => {
      const url = `${import.meta.env.VITE_BACKEND_URL}v1/taxreturns/${taxId}/pdf/${i18n.resolvedLanguage}`;
      try {
        await fetchPdf(url, { hooks: { setLoading, setSuccess } });
      } catch (error) {
        return;
      }
    })();
  };

  return { loading, success, fetchPdf: callFetchPdf };
};
