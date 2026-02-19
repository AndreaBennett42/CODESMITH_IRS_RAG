import { GridContainer } from '@trussworks/react-uswds';
import { BareContentDisplay } from '../ContentDisplay/ContentDisplay.js';
import styles from './PrivacyBanner.module.scss';

const PrivacyBanner = () => {
  return (
    <aside className={styles.privacyBanner}>
      <GridContainer className='usa-prose'>
        <BareContentDisplay i18nKey='privacyBanner' collectionId={null}></BareContentDisplay>
      </GridContainer>
    </aside>
  );
};

export default PrivacyBanner;
