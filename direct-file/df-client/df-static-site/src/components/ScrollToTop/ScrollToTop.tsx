import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

interface ScrollToTopProps {
  focusPoint: React.RefObject<HTMLElement>;
}

/**
 * Most of the time all you need is to “scroll to the top”
 * because you have a long content page, that when navigated to,
 * stays scrolled down. This is straightforward to handle
 * with a <ScrollToTop> component that will scroll the window up
 * on every navigation.
 * https://v5.reactrouter.com/web/guides/scroll-restoration
 */
export default function ScrollToTop({ focusPoint }: ScrollToTopProps) {
  const { pathname } = useLocation();

  useEffect(() => {
    focusPoint?.current?.focus();
    window.scrollTo({
      top: 0,
      behavior: `instant`,
    });
  }, [pathname, focusPoint]);

  return null;
}
