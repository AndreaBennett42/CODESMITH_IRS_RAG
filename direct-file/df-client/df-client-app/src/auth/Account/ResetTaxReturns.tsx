import { useTranslation } from 'react-i18next';
import { save } from '../../hooks/useApiHook.js';
import { useCallback, useContext, useState } from 'react';
import ConfirmationModal from '../../components/ConfirmationModal/ConfirmationModal.js';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { ConcretePath } from '@irs/js-factgraph-scala';
import Heading from '../../components/Heading.js';
import { BackButton } from '../../screens/ScreenHeader.js';
import ContentDisplay from '../../components/ContentDisplay/ContentDisplay.js';
import { Icon } from '@trussworks/react-uswds';
import {
  SetSystemAlertConfig,
  SystemAlertKey,
  useSystemAlertContext,
} from '../../context/SystemAlertContext/SystemAlertContext.js';
import SystemAlertAggregator from '../../components/SystemAlertAggregator/SystemAlertAggregator.js';
import { isPostResubmissionDeadline } from '../../utils/dateUtils.js';
import { v4 as uuidv4 } from 'uuid';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';

const ResetTaxReturns = () => {
  const { setSystemAlert, deleteSystemAlert } = useSystemAlertContext();
  const { currentTaxReturnId } = useContext(TaxReturnsContext);
  const [hasSuccessfulReset, setSuccessfulReset] = useState<boolean>(false);
  const { factGraph } = useFactGraph();
  const now = new Date();
  const isAfterResubmissionDeadline = isPostResubmissionDeadline(now);
  const resetI18nKey = !isAfterResubmissionDeadline ? `reset` : `remove`;

  const handleReset = useCallback(async () => {
    const email = factGraph.get(`/email` as ConcretePath).get.toString();
    const { area, group, serial } = factGraph.get(`/primaryFiler/tin` as ConcretePath).get;
    const primaryFilerId = uuidv4();
    const secondaryFilerId = uuidv4();
    const url = `${import.meta.env.VITE_BACKEND_URL}v1/taxreturns/${currentTaxReturnId}`;
    const alertKey = SystemAlertKey.RESET;

    try {
      await save(url, {
        body: {
          facts: {
            '/email': {
              $type: `gov.irs.factgraph.persisters.EmailAddressWrapper`,
              item: { email },
            },
            [`/filers/#${secondaryFilerId}/isPrimaryFiler`]: {
              $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
              item: false,
            },
            [`/filers/#${primaryFilerId}/isPrimaryFiler`]: {
              $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
              item: true,
            },
            [`/filers/#${primaryFilerId}/tin`]: {
              $type: `gov.irs.factgraph.persisters.TinWrapper`,
              item: { area, group, serial },
            },
            '/filers': {
              $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
              item: { items: [primaryFilerId, secondaryFilerId] },
            },
          },
        },
      });
      deleteSystemAlert(alertKey);
      setSuccessfulReset(true);
    } catch (err) {
      const config: SetSystemAlertConfig = {
        type: `error`,
        i18nKey: `generic.serverError`,
      };
      setSystemAlert(alertKey, config);
    }
  }, [deleteSystemAlert, factGraph, setSystemAlert, currentTaxReturnId]);

  const { t } = useTranslation(`translation`);
  if (hasSuccessfulReset) {
    return (
      <div>
        <SystemAlertAggregator />
        <Heading i18nKey={`/heading/account/${resetI18nKey}-success`} collectionId={null} />
        <ContentDisplay i18nKey={`account.${resetI18nKey}Success`} />
        {/* Need to use anchor tag to force browser refresh */}
        <a href={`${import.meta.env.VITE_PUBLIC_PATH}/home`} className='usa-button'>
          {t(`button.dashboard`)}
          <Icon.NavigateNext size={3} className='usa-button__icon-right' aria-hidden='true' />
        </a>
      </div>
    );
  }

  return (
    <>
      <BackButton />
      <SystemAlertAggregator />
      <Heading i18nKey={`/heading/account/${resetI18nKey}-return`} collectionId={null} />
      <ContentDisplay i18nKey={`account.${resetI18nKey}`} />
      <ConfirmationModal
        i18nKey={`fields./resetReturn.${resetI18nKey}`}
        handleConfirm={handleReset}
        collectionId={null}
        destructiveAction
        modalOpenerClasses='usa-button usa-button--outline usa-button--secondary'
      />
    </>
  );
};

export default ResetTaxReturns;
