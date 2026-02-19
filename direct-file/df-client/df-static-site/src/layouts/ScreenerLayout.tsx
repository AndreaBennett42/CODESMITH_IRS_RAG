import { useRef, ReactNode } from 'react';
import { GridContainer, Grid } from '@trussworks/react-uswds';

import { ScrollToTop } from '../components/index.js';

const ScreenerLayout = ({ page }: { page: () => ReactNode }) => {
  const mainRef = useRef<HTMLElement>(null);

  return (
    <div className='main-content-wrapper'>
      <GridContainer className='screen'>
        <Grid row>
          <Grid col>
            <main id='main' ref={mainRef} tabIndex={-1}>
              <ScrollToTop focusPoint={mainRef} />
              {page()}
            </main>
          </Grid>
        </Grid>
      </GridContainer>
    </div>
  );
};

export default ScreenerLayout;
