import { RefObject } from 'react';

export const numericOnly = (s: string): string => {
  return s.replace(/\D/g, ``);
};

export const alphaCharOnly = (s: string): string => {
  return s.replace(/[^a-zA-Z]+/g, ``);
};

export const alphaNumericCharOnly = (s: string): string => {
  return s.replace(/[^a-zA-Z0-9]+/g, ``);
};

export const numericDashesOnly = (s: string): string => {
  return s.replace(/[^0-9-]+/g, ``);
};

export const numericFloatOnly = (s: string): string => {
  return s.replace(/[^0-9.]+/g, ``);
};

export const convertFlowPathToTransKey = (path: string) => {
  const origPath = path.replace(/\//g, ` `).trim();
  return origPath.replace(/\s/g, `.`);
};

export const convertFactToTransKey = (path: string) => {
  return path.replace(`/`, ``);
};

export const isNavigator = typeof navigator !== `undefined`;

export const stripNonNumeric = (str: string) => str.replace(/[^0-9]/g, ``);

/**
 * Scroll to top of screen and reset focus to the browser's default.
 *
 * To get `prefersReducedMotion` call `usePrefersReducedMotion()`
 */
export const scrollToTop = (prefersReducedMotion: boolean) => {
  const behavior = prefersReducedMotion ? `instant` : `smooth`;
  if (document.activeElement instanceof HTMLElement) {
    document.activeElement.blur();
  }
  window.scrollTo({ top: 0, behavior: behavior });
};

/**
 * Scroll to top of screen header and set focus to an element there.

 */
export const scrollToScreenHeader = () => {
  const topElement = document.getElementById(`main`);

  // Move the focus to the top of the page without selecting a specific element.
  // Tab will move the user to the first element, often an error or alert summary.
  if (topElement) {
    topElement.focus();
  } else if (document.activeElement instanceof HTMLElement) {
    document.activeElement.blur();
  }

  window.scrollTo({ top: 0 });
};

type ScrollableRef = RefObject<HTMLElement> | HTMLElement | null | undefined;

/**
 * Scroll to top of a given ref or element and put focus on it.
 */
export const scrollToRef = (ref: ScrollableRef) => {
  // eslint-disable-next-line eqeqeq
  if (ref && typeof ref == `object`) {
    if (`current` in ref) {
      ref.current?.focus();
      ref.current?.scrollIntoView();
    } else if (ref instanceof HTMLElement) {
      ref.focus();
      ref.scrollIntoView();
    }
  }
};

export const setSmoothScroll = (enabled: boolean) => {
  if (enabled) {
    document.documentElement.classList.add(`enable-smooth-scroll`);
  } else {
    document.documentElement.classList.remove(`enable-smooth-scroll`);
  }
};
