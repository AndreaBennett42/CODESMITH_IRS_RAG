import { render } from '@testing-library/react';

import Income from '../Income.js';
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

describe(`Income page`, () => {
  test(`matches snapshot`, () => {
    const { asFragment } = render(wrapComponent(<Income />, { route: `/income` }));
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
          aria-label="progress"
          class="usa-step-indicator usa-step-indicator--no-labels"
          data-testid="step-indicator"
        >
          <ol
            class="usa-step-indicator__segments"
          >
            <li
              class="usa-step-indicator__segment usa-step-indicator__segment--complete"
            >
              <span
                class="usa-step-indicator__segment-label"
              >
                pages.ScreenerState.stepIndicator.label 
                <span
                  class="usa-sr-only"
                >
                  completed
                </span>
              </span>
            </li>
            <li
              aria-current="true"
              class="usa-step-indicator__segment usa-step-indicator__segment--current"
            >
              <span
                class="usa-step-indicator__segment-label"
              >
                pages.ScreenerIncome.stepIndicator.label 
              </span>
            </li>
            <li
              class="usa-step-indicator__segment"
            >
              <span
                class="usa-step-indicator__segment-label"
              >
                pages.ScreenerSavingsAndRetirement.stepIndicator.label 
                <span
                  class="usa-sr-only"
                >
                  not completed
                </span>
              </span>
            </li>
            <li
              class="usa-step-indicator__segment"
            >
              <span
                class="usa-step-indicator__segment-label"
              >
                pages.ScreenerDeductions.stepIndicator.label 
                <span
                  class="usa-sr-only"
                >
                  not completed
                </span>
              </span>
            </li>
            <li
              class="usa-step-indicator__segment"
            >
              <span
                class="usa-step-indicator__segment-label"
              >
                pages.ScreenerCredits.stepIndicator.label 
                <span
                  class="usa-sr-only"
                >
                  not completed
                </span>
              </span>
            </li>
          </ol>
          <div
            class="usa-step-indicator__header"
          >
            <h1
              class="usa-step-indicator__heading"
            >
              <span
                class="usa-step-indicator__heading-counter"
              >
                <span
                  class="usa-sr-only"
                  data-testid="step-text"
                >
                  components.stepIndicator.stepText
                </span>
                <span
                  class="usa-step-indicator__current-step"
                >
                  2
                </span>
                 
                <span
                  class="usa-step-indicator__total-steps"
                >
                  components.stepIndicator.ofText 5
                </span>
                 
              </span>
              <span
                class="usa-step-indicator__heading-text"
              >
                pages.ScreenerIncome.stepIndicator.label
              </span>
            </h1>
          </div>
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
              pages.ScreenerIncome.heading
            </h2>
          </div>
          <div
            class="margin-top-3"
          >
            <ul
              class="usa-icon-list"
              data-testid="iconList"
            >
              <li
                class="usa-icon-list__item"
                data-testid="iconListItem"
              >
                <div
                  aria-hidden="true"
                  class="text-green usa-icon-list__icon"
                  data-testid="iconListIcon"
                >
                  <svg
                    class="usa-icon"
                    focusable="false"
                    height="1em"
                    role="img"
                    viewBox="0 0 24 24"
                    width="1em"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
                    />
                  </svg>
                </div>
                <div
                  class="usa-icon-list__content"
                  data-testid="iconListContent"
                />
              </li>
              <li
                class="usa-icon-list__item"
                data-testid="iconListItem"
              >
                <div
                  aria-hidden="true"
                  class="text-green usa-icon-list__icon"
                  data-testid="iconListIcon"
                >
                  <svg
                    class="usa-icon"
                    focusable="false"
                    height="1em"
                    role="img"
                    viewBox="0 0 24 24"
                    width="1em"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
                    />
                  </svg>
                </div>
                <div
                  class="usa-icon-list__content"
                  data-testid="iconListContent"
                />
              </li>
              <li
                class="usa-icon-list__item"
                data-testid="iconListItem"
              >
                <div
                  aria-hidden="true"
                  class="text-green usa-icon-list__icon"
                  data-testid="iconListIcon"
                >
                  <svg
                    class="usa-icon"
                    focusable="false"
                    height="1em"
                    role="img"
                    viewBox="0 0 24 24"
                    width="1em"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
                    />
                  </svg>
                </div>
                <div
                  class="usa-icon-list__content"
                  data-testid="iconListContent"
                />
              </li>
              <li
                class="usa-icon-list__item"
                data-testid="iconListItem"
              >
                <div
                  aria-hidden="true"
                  class="text-green usa-icon-list__icon"
                  data-testid="iconListIcon"
                >
                  <svg
                    class="usa-icon"
                    focusable="false"
                    height="1em"
                    role="img"
                    viewBox="0 0 24 24"
                    width="1em"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
                    />
                  </svg>
                </div>
                <div
                  class="usa-icon-list__content"
                  data-testid="iconListContent"
                />
              </li>
              <li
                class="usa-icon-list__item"
                data-testid="iconListItem"
              >
                <div
                  aria-hidden="true"
                  class="text-green usa-icon-list__icon"
                  data-testid="iconListIcon"
                >
                  <svg
                    class="usa-icon"
                    focusable="false"
                    height="1em"
                    role="img"
                    viewBox="0 0 24 24"
                    width="1em"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
                    />
                  </svg>
                </div>
                <div
                  class="usa-icon-list__content"
                  data-testid="iconListContent"
                />
              </li>
              <li
                class="usa-icon-list__item"
                data-testid="iconListItem"
              >
                <div
                  aria-hidden="true"
                  class="text-green usa-icon-list__icon"
                  data-testid="iconListIcon"
                >
                  <svg
                    class="usa-icon"
                    focusable="false"
                    height="1em"
                    role="img"
                    viewBox="0 0 24 24"
                    width="1em"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
                    />
                  </svg>
                </div>
                <div
                  class="usa-icon-list__content"
                  data-testid="iconListContent"
                />
              </li>
              <li
                class="usa-icon-list__item"
                data-testid="iconListItem"
              >
                <div
                  aria-hidden="true"
                  class="text-green usa-icon-list__icon"
                  data-testid="iconListIcon"
                >
                  <svg
                    class="usa-icon"
                    focusable="false"
                    height="1em"
                    role="img"
                    viewBox="0 0 24 24"
                    width="1em"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"
                    />
                  </svg>
                </div>
                <div
                  class="usa-icon-list__content"
                  data-testid="iconListContent"
                />
              </li>
            </ul>
          </div>
          <div
            class="usa-alert usa-alert--warning usa-alert--validation margin-top-3"
            data-testid="alert"
          >
            <div
              class="usa-alert__body"
            />
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
                  aria-controls="pages.ScreenerIncome.AdditionalLimitations"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerIncome.AdditionalLimitations"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerIncome.AdditionalLimitations"
                hidden=""
                id="pages.ScreenerIncome.AdditionalLimitations"
              />
            </div>
          </div>
        </div>
        <div
          class="margin-y-2"
        >
          <a
            class="usa-button"
            href="/savingsandretirement"
          >
            pages.ScreenerIncome.button.text
          </a>
        </div>
      </DocumentFragment>
    `);
    /* eslint-enable */
  });
});
