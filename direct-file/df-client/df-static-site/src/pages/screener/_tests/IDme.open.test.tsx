import { render } from '@testing-library/react';

import IDme from '../IDme.js';
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

describe(`IDme page`, () => {
  test(`matches snapshot`, () => {
    const { asFragment } = render(wrapComponent(<IDme />, { route: `/idme` }));
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
            href="/credits"
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
              pages.ScreenerIDme.heading
            </h2>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerIDme.accordionText"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerIDme.accordionText"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerIDme.accordionText"
                hidden=""
                id="pages.ScreenerIDme.accordionText"
              />
            </div>
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
            href="/done"
          >
            pages.ScreenerIDme.button.text
          </a>
        </div>
      </DocumentFragment>
    `);
    /* eslint-enable */
  });
});
