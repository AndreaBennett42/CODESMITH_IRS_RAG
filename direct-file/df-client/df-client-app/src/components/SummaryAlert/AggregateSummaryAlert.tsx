import usePrefersReducedMotion from '../../hooks/usePrefersReducedMotion.js';
import { Alert } from '@trussworks/react-uswds';
import { scrollToInvalidElement } from './summaryHelpers.js';
import Translation from '../Translation/index.js';
import { useTranslation } from 'react-i18next';

export type AggregateSummaryAlertProps = {
  summaryErrorSections: { i18nKey: string; path: string; isIncomplete: boolean }[];
  summaryWarningSections: { i18nKey: string; path: string }[];
  refs: React.MutableRefObject<Map<string, React.MutableRefObject<HTMLElement>>>;
  headingLevel?: 'h1' | 'h2' | 'h3';
  collectionName: string;
  collectionId?: string | null;
};

const AggregateSummaryAlert = ({
  summaryErrorSections,
  summaryWarningSections,
  refs,
  headingLevel,
  collectionName = ``,
  collectionId = null,
}: AggregateSummaryAlertProps) => {
  // TODO: incompletions:
  //       - If there are no error or warning sections, and instead ONLY incompletions, display "error" type alert with
  //         specific content
  const { i18n, t } = useTranslation();
  const prefersReducedMotion = usePrefersReducedMotion();
  const alertHeadingLevel = headingLevel ? headingLevel : `h1`;

  const hasErrors = summaryErrorSections.length > 0;
  const allErrorsAreIncompletions = !summaryErrorSections.find((err) => !err.isIncomplete);

  const type = hasErrors ? `error` : `warning`;
  const textType = hasErrors && allErrorsAreIncompletions ? `incomplete` : type;

  const hasCollectionSpecificContent =
    collectionName && i18n.exists(`aggregateSummaryAlert.${textType}.collectionItem.${collectionName}`);

  // There may be multiple alerts (e.g. warnings and errors) in the same section. We only need jump links to *unique*
  // paths
  const uniqueSections = [...summaryErrorSections, ...summaryWarningSections].filter(
    (sectionToFilter, sectionToFilterIndex, nonUniqueSections) =>
      sectionToFilterIndex ===
      nonUniqueSections.findIndex(
        (potentiallyNonUniqueSection) => potentiallyNonUniqueSection.path === sectionToFilter.path
      )
  );

  return (
    <Alert
      type={type}
      heading={t(`aggregateSummaryAlert.${textType}.heading`)}
      headingLevel={alertHeadingLevel}
      validation
      data-testid={`aggregate-summary-alert`}
    >
      <ul>
        {uniqueSections.map((s, index) => {
          /* 
            Adding a second key with "__errorLinkTextOverride" allows us to use different
            link text that appears in the alert, since we can't pass variables  
            
            Example:
              coverage-and-contributions: "{{/filers/ * /firstName}'s HSA coverage and contributions"
              coverage-and-contributions__errorLinkTextOverride: "HSA coverage and contributions
          */
          const overrideKeyExists = i18n.exists(`${s.i18nKey}__errorLinkTextOverride`);
          const linkTextKey = overrideKeyExists ? `${s.i18nKey}__errorLinkTextOverride` : s.i18nKey;
          return (
            <li key={index}>
              <a href={``} data-path={s.path} onClick={(e) => scrollToInvalidElement(e, refs, prefersReducedMotion)}>
                <Translation i18nKey={linkTextKey} collectionId={null} />
              </a>
            </li>
          );
        })}
      </ul>
      {hasCollectionSpecificContent && (
        <p>
          <Translation
            i18nKey={`aggregateSummaryAlert.${textType}.collectionItem.${collectionName}`}
            collectionId={collectionId}
          />
        </p>
      )}
    </Alert>
  );
};

export default AggregateSummaryAlert;
