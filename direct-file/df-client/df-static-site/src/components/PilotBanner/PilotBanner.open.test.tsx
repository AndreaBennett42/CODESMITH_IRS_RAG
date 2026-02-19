import { render } from '@testing-library/react';

import PilotBanner from './PilotBanner.js';
import { wrapComponent } from '../../test/helpers.js';
import { PhaseProvider } from '../../layouts/Providers.js';

const mockDate: Date = vi.hoisted(() => new Date(2024, 1, 22, 0, 0, 0));
vi.mock(`../../constants.js`, async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...(actual as object),
    __esModule: true,
    TODAY: mockDate,
  };
});

describe(`Pilot banner`, () => {
  test(`matches snapshot during open filing season`, async () => {
    expect(mockDate.toLocaleString(`default`, { month: `long` })).toBe(`February`);
    const { asFragment } = render(
      wrapComponent(
        <PhaseProvider>
          <PilotBanner />
        </PhaseProvider>,
        { route: `/` }
      )
    );
    const fragment = asFragment();
    /* eslint-disable */
    expect(fragment).toMatchInlineSnapshot(`<DocumentFragment />`);
    /* eslint-enable */
  });
});
