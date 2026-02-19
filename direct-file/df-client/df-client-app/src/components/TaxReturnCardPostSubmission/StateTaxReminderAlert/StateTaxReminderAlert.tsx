import { StateProfile } from '../../../types/StateProfile.js';
import { useTranslation } from 'react-i18next';
import DFAlert from '../../Alert/DFAlert.js';
import { FILING_DEADLINE } from '../../../constants/taxConstants.js';
import { formatAsContentDate, isPostFederalFilingDeadline } from '../../../utils/dateUtils.js';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import useFact from '../../../hooks/useFact.js';
import { Path } from '../../../flow/Path.js';
import { REF_LOCATION, REF_LOCATION_VALUE } from '../../../constants/pageConstants.js';
import Translation from '../../Translation/index.js';

export type StateTaxReminderAlertProps = {
  stateProfile: StateProfile;
  taxYear: number;
  scopedStateDoesNotHavePersonalIncomeTax: boolean;
};

const StateTaxReminderAlert = ({ stateProfile, taxYear }: StateTaxReminderAlertProps) => {
  const { t, i18n } = useTranslation(`translation`);
  const { stateCode, taxSystemName: stateTaxSystemName, landingUrl } = stateProfile || {};
  const stateTaxesId = `state-taxes`;
  const stateName = t(`enums.statesAndProvinces.${stateCode}`);
  const filingDeadline = formatAsContentDate(FILING_DEADLINE, i18n);
  const context = {
    stateName,
    stateTaxSystemName,
    taxYear,
    filingDeadline,
  };

  const [stateCanTransferData] = useFact<boolean>(Path.concretePath(`/stateCanTransferData`, null));
  const isAfterFederalFilingDeadline = isPostFederalFilingDeadline(new Date());

  // set i18nKey
  let i18nKey;
  let linkI18nKey;
  // eslint-disable-next-line eqeqeq
  if (stateCode == `WA`) {
    i18nKey = `taxReturnCard.stateFilingReminder.washington`;
    linkI18nKey = `${i18nKey}.linkText`;
  } else if (!stateCanTransferData) {
    const baseI18nKey = `taxReturnCard.stateFilingReminder.fileYourStateTaxesWithoutTransfer`;
    i18nKey = isAfterFederalFilingDeadline ? `${baseI18nKey}.afterFilingDeadline` : baseI18nKey;
    linkI18nKey = `${baseI18nKey}.linkText`;
  } else {
    const baseI18nKey = `taxReturnCard.stateFilingReminder.fileYourStateTaxesWithTransfer`;
    i18nKey = isAfterFederalFilingDeadline ? `${baseI18nKey}.afterFilingDeadline` : baseI18nKey;
    linkI18nKey = `${baseI18nKey}.linkText`;
  }

  const addQueryParams = (landingUrl: string) => {
    const asUrl = new URL(landingUrl);
    asUrl.searchParams.append(REF_LOCATION, REF_LOCATION_VALUE.SUBMISSION);
    return asUrl.toString();
  };
  const getLandingUrl = () => {
    if (stateCanTransferData) {
      return `/file-your-state-taxes`;
    }
    return addQueryParams(landingUrl);
  };

  const stateTaxesLink = <a href={`#${stateTaxesId}`} />;
  return (
    <DFAlert
      i18nKey={i18nKey}
      type='warning'
      headingLevel='h2'
      className='margin-bottom-2'
      collectionId={null}
      context={context}
      additionalComponents={{ stateTaxesLink }}
    >
      <div>
        <CommonLinkRenderer className='usa-button width-full margin-top-2 margin-bottom-1' url={getLandingUrl()}>
          <Translation i18nKey={linkI18nKey} collectionId={null} context={context} />
        </CommonLinkRenderer>
      </div>
    </DFAlert>
  );
};

export default StateTaxReminderAlert;
