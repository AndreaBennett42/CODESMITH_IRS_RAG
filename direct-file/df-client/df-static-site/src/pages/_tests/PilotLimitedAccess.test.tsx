import { render } from '@testing-library/react';

import PilotLimitedAccess from '../PilotLimitedAccess.js';
import { wrapComponent } from '../../test/helpers.js';

describe(`Intro page`, () => {
  test(`matches snapshot`, () => {
    const { asFragment } = render(wrapComponent(<PilotLimitedAccess />));
    const fragment = asFragment();
    /* eslint-disable */
    expect(fragment).toMatchInlineSnapshot(`
      <DocumentFragment>
        <div
          class="margin-top-1 margin-bottom-3"
        >
          <a
            href="/done"
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
            <h1>
              pages.PilotLimitedAccess.heading
            </h1>
          </div>
        </div>
      </DocumentFragment>
    `);
    /* eslint-enable */
  });
});
