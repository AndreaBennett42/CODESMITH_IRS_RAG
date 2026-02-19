import classNames from 'classnames';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import { Button, Grid, ModalRef } from '@trussworks/react-uswds';
import { useTranslation } from 'react-i18next';
import { TaxReturn } from '../../types/core.js';
import { StateProfile } from '../../types/StateProfile.js';
import { isReturnTransferEnabled, REF_LOCATION, REF_LOCATION_VALUE } from '../../constants/pageConstants.js';
import useFact from '../../hooks/useFact.js';
import { Path } from '../../flow/Path.js';
import { FILING_DEADLINE } from '../../constants/taxConstants.js';
import { MouseEventHandler, useMemo, useRef } from 'react';
import DFAlert from '../Alert/DFAlert.js';
import FileYourStateTaxesModal from '../FileYourStateTaxesModal/FileYourStateTaxesModal.js';
import Translation from '../Translation/index.js';
import { formatAsContentDate, isPostFederalFilingDeadline } from '../../utils/dateUtils.js';

export type StateTaxesCardProps = {
  id: string;
  taxYear: TaxReturn[`taxYear`];
  stateProfile: StateProfile;
  stateCanTransferData: boolean;
  returnWasRejected: boolean;
};

export const buildLandingUrl = (urlString: string) => {
  const url = new URL(urlString);
  url.searchParams.append(REF_LOCATION, REF_LOCATION_VALUE.HOME);
  return url;
};

const StateTaxesCard = ({ id, stateProfile, returnWasRejected, taxYear }: StateTaxesCardProps) => {
  const { t, i18n } = useTranslation(`translation`);

  const { stateCode, taxSystemName, landingUrl } = stateProfile;
  const stateName = t(`enums.statesAndProvinces.${stateCode}`);

  const isAfterFilingDeadline = isPostFederalFilingDeadline(new Date());

  // May want to consider using stateProfile instead
  const [stateCanTransferData] = useFact(Path.concretePath(`/stateCanTransferData`, null));
  const isStateWashington = stateCode === `WA`;

  const bodyI18nKey = useMemo((): string => {
    if (isStateWashington) {
      return `stateTaxesCard.washingtonBody`;
    }

    if (returnWasRejected) {
      return stateCanTransferData
        ? `stateTaxesCard.rejectedGenericBody`
        : `stateTaxesCard.rejectedGenericBodyNoStateIntegration`;
    }

    const transferDisabled = !isReturnTransferEnabled();

    if (stateCanTransferData && !transferDisabled) {
      return `stateTaxesCard.genericBody`;
    }
    return `stateTaxesCard.genericBodyTransferDisabled`;
  }, [isStateWashington, returnWasRejected, stateCanTransferData]);

  const getButtonTextI18nKey = () => {
    if (isStateWashington) {
      return `stateTaxesCard.washingtonButtonText`;
    }

    if (stateCanTransferData) {
      return `stateTaxesCard.getStartedButtonText`;
    }

    return `stateTaxesCard.startYourStateTaxesButtonText`;
  };

  const filingDeadline = formatAsContentDate(FILING_DEADLINE, i18n);

  const stateTaxToolLandingUrl = buildLandingUrl(landingUrl).toString();

  const fileYourStateTaxesUrl = stateCanTransferData ? `/file-your-state-taxes` : stateTaxToolLandingUrl;

  const heading = t(`stateTaxesCard.heading`, { taxYear });
  const modalButtonText = t(`stateTaxesCard.modalButtonText`);

  const buttonClassName = classNames(`usa-button`, `width-full`);

  const context = {
    stateName,
    stateTaxSystemName: taxSystemName,
    filingDeadline,
    taxYear,
  };

  const fileYourStateTaxesModalRef = useRef<ModalRef>(null);
  const toggleModal: MouseEventHandler<HTMLButtonElement> = (e) => {
    fileYourStateTaxesModalRef.current?.toggleModal(e, true);
  };

  return (
    <>
      <FileYourStateTaxesModal modalRef={fileYourStateTaxesModalRef} stateProfile={stateProfile} />
      <Grid id={id} col={12} className='margin-top-3 border-base-lighter border-2px' data-testid='state-taxes-card'>
        <h2 className='margin-0 bg-base-lightest padding-2'>{heading}</h2>
        <div className='usa-prose padding-205'>
          {!returnWasRejected && !isStateWashington && (
            <DFAlert
              i18nKey={`stateTaxesCard.alertHeading${
                isAfterFilingDeadline ? `.afterFilingDeadline` : `.beforeFilingDeadline`
              }`}
              type='warning'
              headingLevel='h2'
              collectionId={null}
              context={context}
              data-testid='state-taxes-card-alert'
            />
          )}
          <p data-testid='state-taxes-card-body'>
            <Translation i18nKey={bodyI18nKey} collectionId={null} context={context} />
          </p>
          {!returnWasRejected || isStateWashington ? (
            <CommonLinkRenderer className={buttonClassName} url={fileYourStateTaxesUrl}>
              <Translation i18nKey={getButtonTextI18nKey()} collectionId={null} context={context} />
            </CommonLinkRenderer>
          ) : (
            <Button type='button' unstyled onClick={toggleModal}>
              {modalButtonText}
            </Button>
          )}
        </div>
      </Grid>
    </>
  );
};

export default StateTaxesCard;
