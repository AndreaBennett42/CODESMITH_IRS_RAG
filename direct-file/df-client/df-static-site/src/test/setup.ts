import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
// needed for expect matcher functions
import '@testing-library/jest-dom';
import { i18n } from 'i18next';

vi.mock(`i18next`, async () => {
  const actual = (await vi.importActual(`i18next`)) as i18n;
  return { ...actual, t: (key: string) => key };
});

// runs a cleanup after each test case (e.g. clearing jsdom)
afterEach(() => {
  cleanup();
});
