import { cleanup, render, screen } from '@testing-library/react';
import { afterEach } from 'vitest';

afterEach(() => {
  cleanup();
});

const customRender = (ui: React.ReactElement, options = {}) =>
  render(ui, {
    // wrap provider(s) here if needed
    wrapper: ({ children }) => children,
    ...options,
  });

// This finds the inner most html element that has the specified text.
// This prevents the issue where testing library finds multiple elements and results in an error
export const getByTextWithTags = (text: string) => {
  return screen.getByText((_, element) => {
    const hasText = (element: HTMLElement) => element.textContent === text;
    const children = Array.from(element?.children || []);
    const childrenDontHaveText = children.every((child) => !hasText(child as HTMLElement));
    return hasText(element as HTMLElement) && childrenDontHaveText;
  });
};

export * from '@testing-library/react';
export { default as userEvent } from '@testing-library/user-event';
// override render export
export { customRender as customRender };
