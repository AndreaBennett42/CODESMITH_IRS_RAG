import usePrefersReducedMotion from '../../hooks/usePrefersReducedMotion.js';
import { Alert } from '@trussworks/react-uswds';
import { scrollToInvalidElement } from './summaryHelpers.js';
import Translation from '../Translation/index.js';
import { useTranslation } from 'react-i18next';
import { DFAlertProps } from '../Alert/DFAlert.js';
import { Link } from 'react-router-dom';

export type SummaryAlertProps = {
  sections: { i18nKey: string; path: string; isIncomplete?: boolean }[];
  refs: React.MutableRefObject<Map<string, React.MutableRefObject<HTMLElement>>>;
  headingLevel?: 'h1' | 'h2' | 'h3';
  type: Exclude<DFAlertProps['type'], 'success' | 'info'>;
};

const SummaryAlert = ({ sections, refs, headingLevel, type }: SummaryAlertProps) => {
  const { t } = useTranslation();
  const prefersReducedMotion = usePrefersReducedMotion();
  const alertHeadingLevel = headingLevel ? headingLevel : `h1`;
  const allAreIncompletions = sections.every((section) => section.isIncomplete);
  // TO DO have one incomplete and one error

  const alertType = allAreIncompletions ? `incomplete` : type;

  return (
    <Alert
      type={type}
      heading={t(`summaryAlert.${alertType}.heading`, { count: sections.length })}
      headingLevel={alertHeadingLevel}
      validation
      data-testid={`${type}-summary-alert`}
    >
      <ul>
        {sections.map((s, index) => {
          return (
            <li key={index}>
              {s.isIncomplete ? (
                <Link to={s.path}>
                  <Translation i18nKey={s.i18nKey} collectionId={null} />
                </Link>
              ) : (
                <a href={``} data-path={s.path} onClick={(e) => scrollToInvalidElement(e, refs, prefersReducedMotion)}>
                  <Translation i18nKey={s.i18nKey} collectionId={null} />
                </a>
              )}
            </li>
          );
        })}
      </ul>
    </Alert>
  );
};

export default SummaryAlert;
