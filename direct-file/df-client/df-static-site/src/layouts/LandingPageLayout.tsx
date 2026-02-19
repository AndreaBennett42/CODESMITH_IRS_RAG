import { useRef, ReactNode } from 'react';

import { ScrollToTop } from '../components/index.js';

const LandingPageLayout = ({ page }: { page: () => ReactNode }) => {
  const mainRef = useRef<HTMLElement>(null);

  return (
    <div id='splash'>
      <main id='main' ref={mainRef}>
        <ScrollToTop focusPoint={mainRef} />
        {page()}
      </main>
    </div>
  );
};

export default LandingPageLayout;
