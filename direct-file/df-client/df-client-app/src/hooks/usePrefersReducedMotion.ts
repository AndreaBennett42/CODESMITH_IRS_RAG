import { useEffect, useState } from 'react';

/**
 * Rapid motion can be disorienting to some segments of the population, who can note their
 * preference for reduced motion in their browser. This hook detects that preference.
 *
 * See https://www.a11yproject.com/posts/understanding-vestibular-disorders/ for a primer on reduced motion
 * and this CSS selector
 */
export default function usePrefersReducedMotion() {
  const [prefersReducedMotion, setPrefersReducedMotion] = useState<boolean>(false);
  const onQueryChange = ({ matches }: { matches: boolean }) => setPrefersReducedMotion(matches);

  useEffect(() => {
    // If the browser does not support matchMedia, assume they prefer reduced motion
    if (!window.matchMedia) setPrefersReducedMotion(true);

    const mediaQuery = window.matchMedia(`(prefers-reduced-motion: reduce)`);

    mediaQuery.addEventListener(`change`, onQueryChange);

    return () => mediaQuery.removeEventListener(`change`, onQueryChange);
  }, []);

  return prefersReducedMotion;
}
