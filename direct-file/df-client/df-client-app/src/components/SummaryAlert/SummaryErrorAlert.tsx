import { Alert } from '@trussworks/react-uswds';
import { useTranslation } from 'react-i18next';
import { Path } from '../../flow/Path.js';
import { ConcretePath } from '@irs/js-factgraph-scala';
import usePrefersReducedMotion from '../../hooks/usePrefersReducedMotion.js';
import Translation from '../Translation/index.js';
import { useEffect, useMemo } from 'react';
import styles from './SummaryErrorAlert.module.scss';
import { scrollToInvalidElement } from './summaryHelpers.js';

interface SummaryErrorAlertProps {
  factValidity: Map<ConcretePath, boolean>;
  factRefs: React.MutableRefObject<Map<string, React.MutableRefObject<HTMLElement>>>;
  id: string;
}

interface SummaryErrorFacts {
  displayName: string | React.JSX.Element;
  alertName: string;
  path: string;
  factPath: string;
}

const SummaryErrorAlert = ({ factValidity, factRefs, id }: SummaryErrorAlertProps) => {
  useEffect(() => document.getElementById(id)?.focus(), [id]);

  const prefersReducedMotion = usePrefersReducedMotion();

  const { t, i18n } = useTranslation(`translation`);

  const invalidPaths: SummaryErrorFacts[] = useMemo(() => {
    const paths: SummaryErrorFacts[] = [];
    // Grab all invalid paths:
    factValidity.forEach((value, concretePath) => {
      const path = Path.fromConcretePath(concretePath);
      const i18nKey = `fields.${path}.name`;
      if (!value) {
        const alert = {
          // At the moment, the enums (radio buttons) don't have a translation keys so we're defaulting to the
          // chooseOption
          displayName: i18n.exists(i18nKey) ? (
            <Translation i18nKey={i18nKey} collectionId={null} />
          ) : (
            t(`enums.messages.chooseOption`)
          ),
          // GA needs a string value for the field name
          alertName: i18n.exists(i18nKey) ? t(i18nKey) : t(`enums.messages.chooseOption`),
          factPath: path,
          path: concretePath,
        };
        paths.push(alert);
      }
    });
    return paths;
  }, [factValidity, t, i18n]);

  return (
    <Alert
      id={id}
      className={styles.summaryErrorAlert}
      role='alert'
      headingLevel='h2'
      heading={t(`summaryErrorAlert.heading`, { count: invalidPaths.length })}
      type={`error`}
      validation={true}
      tabIndex={-1}
    >
      <ul>
        {invalidPaths.map((fact) => {
          return (
            <li key={fact.path}>
              <a
                href={``}
                onClick={(e) => scrollToInvalidElement(e, factRefs, prefersReducedMotion)}
                data-path={fact.path}
              >
                {fact.displayName}
              </a>
            </li>
          );
        })}
      </ul>
    </Alert>
  );
};

SummaryErrorAlert.displayName = `SummaryErrorAlert`;

export default SummaryErrorAlert;
