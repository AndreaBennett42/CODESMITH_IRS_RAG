import { ProcessList } from '@trussworks/react-uswds';
import ChecklistCategory from '../components/checklist/ChecklistCategory/index.js';
import { TaxReturn } from '../types/core.js';
import PageTitle from '../components/PageTitle/index.js';
import { useTranslation } from 'react-i18next';
import { FC, MutableRefObject, useRef } from 'react';
import { useChecklistState } from '../flow/useChecklistState.js';
import SectionsAlertAggregator from '../components/SectionsAlertAggregator/SectionsAlertAggregator.js';
import { extractUniqueAlertSummarySections } from '../misc/aggregatedAlertHelpers.js';
import useTranslatePIIRedacted from '../hooks/useTranslatePIIRedacted.js';
import { ContextHeading } from '../components/ContextHeading/ContextHeading.js';
import ContentDisplay from '../components/ContentDisplay/ContentDisplay.js';
import { useInitializeChecklist } from '../hooks/useInitializeChecklistHook.js';

export interface ChecklistProps {
  taxReturns: TaxReturn[];
}

const Checklist: FC = () => {
  useInitializeChecklist();
  const { t } = useTranslation();
  const { t: tStates } = useTranslation(`translation`, { keyPrefix: `enums.statesAndProvinces` });

  const categoryRefs = useRef(new Map<string, MutableRefObject<HTMLAnchorElement>>());
  const checklistState = useChecklistState();
  const pageContent = checklistState.map((c) => {
    return <ChecklistCategory {...c} categoryRefs={categoryRefs} key={c.route} />;
  });

  const { summaryErrorSections, summaryWarningSections } = extractUniqueAlertSummarySections(checklistState);
  const redacted = useTranslatePIIRedacted(`checklist.header`, true);

  return (
    <>
      <SectionsAlertAggregator
        showStatusAlert
        summaryErrorSections={summaryErrorSections}
        summaryWarningSections={summaryWarningSections}
        refs={categoryRefs}
      />
      <div className='screen__header'>
        <ContextHeading displayOnlyOn='edit' i18nKey='/heading/checklist' collectionId={null} />
        <PageTitle redactedTitle={redacted} large>
          {t(`checklist.header`)}
        </PageTitle>
        <div className='margin-bottom-1'>
          <ContentDisplay i18nKey='preChecklist.review-what-you-need' />
        </div>
      </div>
      <ProcessList>{pageContent}</ProcessList>
    </>
  );
};

export default Checklist;
