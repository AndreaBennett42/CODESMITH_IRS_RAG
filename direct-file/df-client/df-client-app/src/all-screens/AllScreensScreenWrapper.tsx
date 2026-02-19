import { Suspense, lazy, memo } from 'react';
import { ScreenConfig } from '../flow/ScreenConfig.js';
import styles from './AllScreens.module.scss';
import screenStyles from '../components/Screen.module.scss';
import { uuid } from './AllScreensContext.js';
import { CommonTranslation } from 'df-i18n';
import { ModalHeading } from '@trussworks/react-uswds';
import { useTranslation } from 'react-i18next';
import AllScreensScreenHeader from './AllScreensScreenHeader.js';
import ContentDisplay from '../components/ContentDisplay/index.js';
import Translation from '../components/Translation/index.js';
import classNames from 'classnames';
import { calculateScreenStatus } from '../flow/batches.js';

const ALERT_COMPONENT_TYPES = [`DFAlert`, `TaxReturnAlert`, `MefAlert`] as const;

export const AllScreensScreenWrapper = memo(
  ({ screen, hideAlerts, showModals }: { screen: ScreenConfig; hideAlerts: boolean; showModals: boolean }) => {
    const { t, i18n } = useTranslation();

    const Screen = lazy(() => import(`../components/Screen.js`));

    const modalComponents = screen.content.flatMap((component) => {
      if (component.componentName === `DFModal` || component.props.hintKey) {
        const i18nKey =
          component.componentName === `DFModal`
            ? (component.props as { i18nKey: string }).i18nKey
            : component.props.hintKey;

        if (!i18nKey) return [];

        const fullKey = CommonTranslation.getFallbackKey([i18nKey], i18n);
        const modals = t(`${fullKey}.helpText.modals`, { returnObjects: true });
        const modalKeys = Object.keys(modals).filter((key) => key.startsWith(`LinkModal`));
        const isOpen = calculateScreenStatus(component.props.batches).isOpen;

        return modalKeys.map((modalKey) => {
          const uniqueModalId = `${fullKey}-${modalKey}`;
          return (
            <div key={uniqueModalId} className={styles.individualModal}>
              <ModalHeading
                id={`modal-heading-${uniqueModalId}`}
                className={classNames({ [screenStyles.draftContent]: isOpen })}
              >
                <Translation collectionId={uuid} i18nKey={`${fullKey}.helpText.modals.${modalKey}.header`} />
              </ModalHeading>
              <div
                className={classNames(`usa-prose`, { [screenStyles.draftContent]: isOpen })}
                id={`modal-description-${uniqueModalId}`}
              >
                <ContentDisplay
                  collectionId={uuid}
                  i18nKey={`${fullKey}.helpText.modals.${modalKey}`}
                  additionalComponents={{
                    h2: <h2 className='usa-modal__heading' />,
                  }}
                />
              </div>
            </div>
          );
        });
      }
      return [];
    });

    return (
      <div className={styles.screenOuterContainer} key={screen.route} style={{ display: `block` }}>
        <AllScreensScreenHeader screen={screen} />
        <div className={styles.modalContainer}>
          <div className={styles.screenContainer}>
            <Suspense fallback={<div>Loading...</div>}>
              <Screen
                screenRoute={screen.route}
                collectionId={uuid}
                screenContent={screen.content}
                setFactActionPaths={[]}
                gotoNextScreen={() => {}}
                alertAggregatorType={screen.alertAggregatorType}
                componentsToHide={hideAlerts ? new Set(ALERT_COMPONENT_TYPES) : undefined}
                collectionContext={screen.collectionContext}
              />
            </Suspense>
          </div>
          {showModals && modalComponents.length > 0 && <div className={styles.modals}>{modalComponents}</div>}
        </div>
      </div>
    );
  }
);

AllScreensScreenWrapper.displayName = `AllScreensScreenWrapper`;
