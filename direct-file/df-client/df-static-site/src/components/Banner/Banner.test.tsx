import { render, screen } from '@testing-library/react';

import Banner from './Banner.js';

describe(`Banner component`, () => {
  test(`renders without errors`, () => {
    render(<Banner lang='english' />);
    expect(screen.getByText(`An official website of the United States government`)).toBeInTheDocument();
  });
});
