import { render } from '@testing-library/react';

import State from '../State.js';
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

describe(`State page`, () => {
  test(`matches snapshot`, () => {
    const { asFragment } = render(wrapComponent(<State />, { route: `/state` }));
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
            href="/"
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
              aria-current="true"
              class="usa-step-indicator__segment usa-step-indicator__segment--current"
            >
              <span
                class="usa-step-indicator__segment-label"
              >
                pages.ScreenerState.stepIndicator.label 
              </span>
            </li>
            <li
              class="usa-step-indicator__segment"
            >
              <span
                class="usa-step-indicator__segment-label"
              >
                pages.ScreenerIncome.stepIndicator.label 
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
                  1
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
                pages.ScreenerState.stepIndicator.label
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
              pages.ScreenerState.heading
            </h2>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Alaska"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Alaska"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Alaska"
                hidden=""
                id="pages.ScreenerState.Alaska"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Arizona"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Arizona"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Arizona"
                hidden=""
                id="pages.ScreenerState.Arizona"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.California"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.California"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.California"
                hidden=""
                id="pages.ScreenerState.California"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Connecticut"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Connecticut"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Connecticut"
                hidden=""
                id="pages.ScreenerState.Connecticut"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Florida"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Florida"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Florida"
                hidden=""
                id="pages.ScreenerState.Florida"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Idaho"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Idaho"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Idaho"
                hidden=""
                id="pages.ScreenerState.Idaho"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Illinois"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Illinois"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Illinois"
                hidden=""
                id="pages.ScreenerState.Illinois"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Kansas"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Kansas"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Kansas"
                hidden=""
                id="pages.ScreenerState.Kansas"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Maine"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Maine"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Maine"
                hidden=""
                id="pages.ScreenerState.Maine"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Maryland"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Maryland"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Maryland"
                hidden=""
                id="pages.ScreenerState.Maryland"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Massachusetts"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Massachusetts"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Massachusetts"
                hidden=""
                id="pages.ScreenerState.Massachusetts"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Nevada"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Nevada"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Nevada"
                hidden=""
                id="pages.ScreenerState.Nevada"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.New Hampshire"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.New Hampshire"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.New Hampshire"
                hidden=""
                id="pages.ScreenerState.New Hampshire"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.New Jersey"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.New Jersey"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.New Jersey"
                hidden=""
                id="pages.ScreenerState.New Jersey"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.New Mexico"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.New Mexico"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.New Mexico"
                hidden=""
                id="pages.ScreenerState.New Mexico"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.New York"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.New York"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.New York"
                hidden=""
                id="pages.ScreenerState.New York"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.North Carolina"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.North Carolina"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.North Carolina"
                hidden=""
                id="pages.ScreenerState.North Carolina"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Oregon"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Oregon"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Oregon"
                hidden=""
                id="pages.ScreenerState.Oregon"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Pennsylvania"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Pennsylvania"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Pennsylvania"
                hidden=""
                id="pages.ScreenerState.Pennsylvania"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.South Dakota"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.South Dakota"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.South Dakota"
                hidden=""
                id="pages.ScreenerState.South Dakota"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Tennessee"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Tennessee"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Tennessee"
                hidden=""
                id="pages.ScreenerState.Tennessee"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Texas"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Texas"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Texas"
                hidden=""
                id="pages.ScreenerState.Texas"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Washington"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Washington"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Washington"
                hidden=""
                id="pages.ScreenerState.Washington"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Wisconsin"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Wisconsin"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Wisconsin"
                hidden=""
                id="pages.ScreenerState.Wisconsin"
              />
            </div>
          </div>
          <div
            class="margin-top-3"
          >
            <div
              class="usa-accordion usa-accordion--bordered styled-state"
              data-testid="accordion"
            >
              <h3
                class="usa-accordion__heading"
              >
                <button
                  aria-controls="pages.ScreenerState.Wyoming"
                  aria-expanded="false"
                  class="usa-accordion__button"
                  data-testid="accordionButton_pages.ScreenerState.Wyoming"
                  type="button"
                />
              </h3>
              <div
                class="usa-accordion__content usa-prose"
                data-testid="accordionItem_pages.ScreenerState.Wyoming"
                hidden=""
                id="pages.ScreenerState.Wyoming"
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
            href="/income"
          >
            pages.ScreenerState.button.text
          </a>
        </div>
      </DocumentFragment>
    `);
    /* eslint-enable */
  });
});
