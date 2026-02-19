import { DEFAULT_EXCLUDED_CATEGORIES, useChecklistState } from '../../flow/useChecklistState.js';
import Translation from '../Translation/index.js';
import { Link } from 'react-router-dom';
import TaxReturnAlert, { TaxReturnAlertProps } from '../Alert/TaxReturnAlert.js';
import styles from './SectionReview.module.scss';
import { AlertConfigs, MefAlertConfig } from '../../misc/aggregatedAlertHelpers.js';

const renderAggregatedAlert = (i18nKey: string, type: TaxReturnAlertProps['type'], count?: number) => {
  return (
    <TaxReturnAlert className='margin-right-1' i18nKey={i18nKey} type={type} collectionId={null} context={{ count }} />
  );
};

const renderAggregatedAlertCueForSection = (
  taxReturnAlerts: AlertConfigs,
  mefAlerts: AlertConfigs<MefAlertConfig>,
  hasIncompletedCollectionItem: boolean
) => {
  const errors = [...taxReturnAlerts.errors, ...mefAlerts.errors];
  const warnings = [...taxReturnAlerts.warnings, ...mefAlerts.warnings];

  const hasErrors = errors.length > 0;
  const hasWarnings = warnings.length > 0;
  const hasBoth = hasErrors && hasWarnings;

  if (hasBoth) {
    return renderAggregatedAlert(
      `checklist.alerts.checklistSubcategory.both`,
      `error`,
      errors.length + warnings.length
    );
  }
  if (hasErrors) {
    const alertMessage =
      errors.length > 1
        ? `checklist.alerts.checklistSubcategory.error_other`
        : `checklist.alerts.checklistSubcategory.error_one`;
    return renderAggregatedAlert(alertMessage, `error`, errors.length);
  }
  if (hasWarnings) {
    const alertMessage =
      warnings.length > 1
        ? `checklist.alerts.checklistSubcategory.warning_other`
        : `checklist.alerts.checklistSubcategory.warning_one`;
    return renderAggregatedAlert(alertMessage, `warning`, warnings.length);
  }
  if (hasIncompletedCollectionItem) {
    return renderAggregatedAlert(`checklist.alerts.checklistSubcategory.incompletion`, `error`);
  }
};

const SectionReview = () => {
  // We exclude our own section and sign + submit, because they have nought to do with your taxes.
  const checklistState = useChecklistState(new Set([...DEFAULT_EXCLUDED_CATEGORIES, `/flow/complete`]));
  return (
    <ol className={styles.sectionReviewList}>
      {checklistState.map((category) => {
        return (
          <li key={category.route} className={styles.sectionReviewItem}>
            <h3 className={styles.sectionReviewHeading}>
              <Translation i18nKey={`checklist.${category.route}.heading`} collectionId={null} />
            </h3>
            <ul className={styles.subcategoryList}>
              {category.subcategories.map((sc) => {
                if (sc.navigationUrl === undefined) {
                  return null;
                }
                return (
                  <li key={sc.subcategoryRoute} className={styles.subcategoryRow}>
                    <div className={styles.subcategoryRowHeader}>
                      <h4 className={styles.subcategoryRowHeading}>
                        <Translation i18nKey={`checklist.${sc.subcategoryRoute}.heading`} collectionId={null} />
                      </h4>
                      <Link className={styles.subcategoryReviewLink} to={sc.navigationUrl}>
                        <Translation i18nKey={`dataviews.review`} collectionId={null} />
                        <span className='usa-sr-only'>
                          <Translation i18nKey={`checklist.${sc.subcategoryRoute}.heading`} collectionId={null} />
                        </span>
                      </Link>
                    </div>
                    {renderAggregatedAlertCueForSection(
                      sc.alertConfigs,
                      sc.mefAlertConfigs,
                      sc.hasIncompletedCollectionItem
                    )}
                  </li>
                );
              })}
            </ul>
          </li>
        );
      })}
    </ol>
  );
};

export default SectionReview;
