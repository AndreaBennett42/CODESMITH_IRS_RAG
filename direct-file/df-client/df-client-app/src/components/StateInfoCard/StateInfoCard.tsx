import Translation from '../Translation/index.js';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import InfoDisplay from '../InfoDisplay.js';
import { StateProfile } from '../../types/StateProfile.js';
import { REF_LOCATION, REF_LOCATION_VALUE } from '../../constants/pageConstants.js';
import { useTranslation } from 'react-i18next';
import LoadingIndicator from '../LoadingIndicator/LoadingIndicator.js';
import useFetchStateProfile from '../../hooks/useFetchStateProfile.js';

export type StateInfoCardProps = {
  i18nKey: string;
  stateLinki18nKey: string;
  borderBottom?: boolean;
};

export const appendQueryParams = (url: URL) => {
  url.searchParams.append(REF_LOCATION, REF_LOCATION_VALUE.SUBMISSION);
};

const StateInfoCard = ({ i18nKey, stateLinki18nKey, borderBottom }: StateInfoCardProps) => {
  const { t } = useTranslation(`translation`);
  const { stateProfile, isFetching: isLoadingStateProfile } = useFetchStateProfile();

  if (isLoadingStateProfile) {
    return <LoadingIndicator />;
  }

  if (!isLoadingStateProfile && stateProfile) {
    const context = {
      stateName: t(`enums.statesAndProvinces.${stateProfile.stateCode}`),
      stateTaxSystemName: stateProfile.taxSystemName,
    };

    const renderStateProfileInformation = (stateProfile: StateProfile) => {
      const landingUrl = new URL(stateProfile.landingUrl);
      appendQueryParams(landingUrl);

      return (
        <Translation
          i18nKey={`info.${stateLinki18nKey}`}
          collectionId={null}
          context={context}
          components={{
            StateLink: (
              <CommonLinkRenderer url={landingUrl.toString()}>{stateProfile.taxSystemName}</CommonLinkRenderer>
            ),
          }}
        />
      );
    };

    return (
      <div className={`${borderBottom ? `border-bottom` : ``}`}>
        <InfoDisplay i18nKey={i18nKey} collectionId={null} context={context} />
        <p>{stateProfile && renderStateProfileInformation(stateProfile)}</p>
      </div>
    );
  }
};

export default StateInfoCard;
