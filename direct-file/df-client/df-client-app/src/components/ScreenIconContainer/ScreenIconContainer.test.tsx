import { render, isInaccessible } from '@testing-library/react';
import ScreenIconContainer from './ScreenIconContainer.js';

describe(`ScreenIconContainer`, () => {
  test(`aria-hidden defaults to true`, () => {
    const id = `test-id`;
    const { getByTestId, debug } = render(<ScreenIconContainer data-testid={id}>test</ScreenIconContainer>);
    expect(getByTestId(id).getAttribute(`aria-hidden`)).toEqual(`true`);
    debug();

    expect(isInaccessible(getByTestId(id))).toBeTruthy();
  });

  test(`aria-hidden can be set to false`, () => {
    const id = `test-id`;
    const { getByTestId, debug } = render(
      <ScreenIconContainer data-testid={id} aria-hidden={false}>
        test
      </ScreenIconContainer>
    );
    debug();
    expect(getByTestId(id).getAttribute(`aria-hidden`)).toEqual(`false`);
    expect(isInaccessible(getByTestId(id))).toBeFalsy();
  });
});
