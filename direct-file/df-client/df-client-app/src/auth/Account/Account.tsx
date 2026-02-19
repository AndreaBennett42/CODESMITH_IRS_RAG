import { useTranslation } from 'react-i18next';
import { Link as RouterLink } from 'react-router-dom';
import classNames from 'classnames';

import {
  IconList as USAIconList,
  IconListItem,
  IconListContent,
  IconListIcon,
  Icon,
  Link,
} from '@trussworks/react-uswds';
import { clearBrowserStorage } from '../storage.js';
import PageTitle from '../../components/PageTitle/index.js';
import useTranslatePIIRedacted from '../../hooks/useTranslatePIIRedacted.js';
import { useContext, useMemo } from 'react';
import { SubmissionStatusContext } from '../../context/SubmissionStatusContext/SubmissionStatusContext.js';
import { getTaxReturnById, hasBeenSubmitted } from '../../utils/taxReturnUtils.js';
import { CURRENT_TAX_YEAR, FEDERAL_RETURN_STATUS } from '../../constants/taxConstants.js';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';
import SystemAlertAggregator from '../../components/SystemAlertAggregator/SystemAlertAggregator.js';
import { isPostResubmissionDeadline } from '../../utils/dateUtils.js';
import { ConcretePath } from '@irs/js-factgraph-scala';
import { TaxProfileContextOrSpinnerGate } from '../../screens/TaxProfileContextOrSpinnerGate.js';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { PriorYearTaxReturnCard } from '../../components/TaxReturnCard/PriorYearTaxReturnCard.js';

const Account = () => {
  const { currentTaxReturnId } = useContext(TaxReturnsContext);

  const accountPage = (
    <main id='main' tabIndex={-1}>
      <AccountContent />
    </main>
  );

  // If we have a tax return, load the fact graph for it
  return currentTaxReturnId ? (
    <TaxProfileContextOrSpinnerGate>{accountPage}</TaxProfileContextOrSpinnerGate>
  ) : (
    accountPage
  );
};

export const AccountContent = () => {
  const { taxReturns, currentTaxReturnId } = useContext(TaxReturnsContext);
  const { factGraph } = useFactGraph();
  const currentTaxReturn = getTaxReturnById(taxReturns, currentTaxReturnId);
  const hasSubmitted = currentTaxReturn && hasBeenSubmitted(currentTaxReturn);
  const { submissionStatus } = useContext(SubmissionStatusContext);
  const statusIsRejected = submissionStatus && submissionStatus.status === FEDERAL_RETURN_STATUS.REJECTED;
  const resetDisabled = !!(hasSubmitted && !statusIsRejected);

  const { t } = useTranslation(`translation`);
  const redacted = useTranslatePIIRedacted(`account.title`, true);

  const iconClasses = classNames({ 'text-disabled': resetDisabled });
  const taxYear = currentTaxReturn?.taxYear || CURRENT_TAX_YEAR;
  const now = new Date();
  const isAfterResubmissionDeadline = isPostResubmissionDeadline(now);

  const email = useMemo(
    () => (factGraph ? factGraph.get(`/email` as ConcretePath).get.toString() : undefined),
    [factGraph]
  );

  return (
    <>
      <SystemAlertAggregator />
      <PageTitle redactedTitle={redacted}>{t(`account.title`)}</PageTitle>
      <USAIconList>
        <IconListItem key={1}>
          <IconListIcon>
            <Icon.MailOutline aria-hidden='true' />
          </IconListIcon>
          <IconListContent>
            {email && (
              <>
                {email}
                <br />
              </>
            )}
            <Link role='link' href={`https://account.id.me/profile`} rel='noreferrer' target='_blank'>
              {t(`account.profile`)}
            </Link>
          </IconListContent>
        </IconListItem>
        {!!taxReturns.length && (
          <IconListItem key={2}>
            <IconListIcon>
              {!isAfterResubmissionDeadline ? (
                <Icon.Autorenew aria-hidden='true' />
              ) : (
                <Icon.Delete aria-hidden='true' />
              )}
            </IconListIcon>
            <IconListContent className={iconClasses}>
              {resetDisabled ? (
                t(`account.reset.disabled`, { taxYear })
              ) : (
                <RouterLink to='reset'>
                  {!isAfterResubmissionDeadline
                    ? t(`account.reset.reset-btn`, { taxYear })
                    : t(`account.remove.remove-btn`, { taxYear })}
                </RouterLink>
              )}
            </IconListContent>
          </IconListItem>
        )}
        <IconListItem key={3}>
          <IconListIcon>
            <Icon.Logout aria-hidden='true' />
          </IconListIcon>
        </IconListItem>
      </USAIconList>
      <PriorYearTaxReturnCard />
    </>
  );
};

export default Account;
