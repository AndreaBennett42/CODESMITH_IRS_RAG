import { render, screen } from '@testing-library/react';
import { store } from '../redux/store.js';
import '@testing-library/jest-dom/extend-expect';
import AllScreens from './AllScreens.js';
import { initI18n } from '../i18n.js';
import { act } from 'react';

// Rendering All Screens takes a long time.
// We need to increase the default test timeout to
// ensure we load all the screens.
const ALL_SCREENS_RENDER_TEST_TIMEOUT = 100000;

describe(`All Screens`, () => {
  beforeAll(() => {
    initI18n();
    Element.prototype.scrollIntoView = () => {};
  });
  it(`properly boots the store`, () => {
    // Allscreens is easy to refactor to not pull in the redux store. This causes weird issues.
    expect(Object.keys(store.getState()).length).toBeGreaterThan(0);
  });

  it(
    `renders without errors`,
    async () => {
      // If someone adds a component that relies on some fact graph state,
      // this test case may fail. In that case, you'll likely need to either
      // mock the value into `AllScreen#setupFactGraph`, or find another way
      // around it to not break the All Screen Renderer
      await act(() => render(<AllScreens />));
      const headerElement = screen.getByRole(`heading`, { name: /Direct File \| All Screens/i });
      expect(headerElement).toBeInTheDocument();
    },
    ALL_SCREENS_RENDER_TEST_TIMEOUT
  );
});
