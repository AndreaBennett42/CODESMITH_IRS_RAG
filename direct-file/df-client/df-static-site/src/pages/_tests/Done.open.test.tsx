import { render } from '@testing-library/react';

import Done from '../Done.js';
import { wrapComponent } from '../../test/helpers.js';

const mockDate: Date = vi.hoisted(() => new Date(2024, 1, 22, 0, 0, 0));
vi.mock(`../../constants.js`, async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...(actual as object),
    __esModule: true,
    TODAY: mockDate,
  };
});

describe(`Done page`, () => {
  test(`matches snapshot`, () => {
    const { asFragment } = render(wrapComponent(<Done />, { route: `/done` }));
    const fragment = asFragment();
    expect(fragment).toMatchSnapshot();
  });
});
