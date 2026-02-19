import { render } from '@testing-library/react';

import Identification from '../Identification.js';
import { wrapComponent } from '../../../test/helpers.js';

const mockDate: Date = vi.hoisted(() => new Date(2024, 1, 1, 0, 0, 0));
vi.mock(`../../../constants.js`, async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...(actual as object),
    __esModule: true,
    TODAY: mockDate,
  };
});

describe(`Identification page`, () => {
  test(`matches snapshot`, () => {
    const { asFragment } = render(wrapComponent(<Identification />, { route: `/identification` }));
    const fragment = asFragment();
    /* eslint-disable */
    expect(fragment).toMatchInlineSnapshot(`
      <DocumentFragment>
        <header
          aria-label="components.subheader.label"
          class="site-subheader site-subheader--desktop"
        >
          components.subheader.title
        </header>
        <div
          class="margin-top-1 margin-bottom-3"
        >
          <a
            href="/state"
          >
            <svg
              aria-hidden="true"
              class="usa-icon top-05"
              focusable="false"
              height="1em"
              role="img"
              viewBox="0 0 24 24"
              width="1em"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                d="M20 11H7.83l5.59-5.59L12 4l-8 8 8 8 1.41-1.41L7.83 13H20v-2z"
              />
            </svg>
            <span
              class="margin-left-05"
            >
              components.breadcrumbs.back
            </span>
          </a>
        </div>
        <div
          class="usa-prose"
        >
          <div
            aria-atomic="true"
            aria-live="polite"
            aria-relevant="additions"
            role="log"
          >
            <h2
              class="font-body-lg text-normal margin-top-0"
            >
              pages.ScreenerIdentification.heading
            </h2>
          </div>
          <div
            class="usa-alert usa-alert--warning usa-alert--validation margin-top-3"
            data-testid="alert"
          >
            <div
              class="usa-alert__body"
            />
          </div>
        </div>
        <div
          class="margin-y-2"
        >
          <a
            class="usa-button"
            href="/income"
          >
            pages.ScreenerIdentification.button.text
          </a>
        </div>
      </DocumentFragment>
    `);
    /* eslint-enable */
  });
});
