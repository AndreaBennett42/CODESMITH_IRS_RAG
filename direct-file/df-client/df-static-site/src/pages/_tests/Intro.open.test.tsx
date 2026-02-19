import { render } from '@testing-library/react';

import Intro from '../Intro.js';
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

describe(`Intro page`, () => {
  test(`matches snapshot`, () => {
    expect(mockDate.toLocaleString(`default`, { month: `long` })).toBe(`February`);
    const { asFragment } = render(wrapComponent(<Intro />));
    const fragment = asFragment();
    expect(fragment).toMatchSnapshot();
  });
});
