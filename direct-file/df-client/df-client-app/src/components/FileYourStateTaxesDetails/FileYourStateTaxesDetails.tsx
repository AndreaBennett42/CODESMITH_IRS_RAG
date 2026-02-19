import { useContext } from 'react';
import { useTranslation } from 'react-i18next';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import { TaxReturnsContext } from '../../context/TaxReturnsContext.js';
import { getCurrentTaxYearReturn } from '../../utils/taxReturnUtils.js';
import useFetchStateProfile from '../../hooks/useFetchStateProfile.js';
import LoadingIndicator from '../LoadingIndicator/LoadingIndicator.js';
import Translation from '../Translation/index.js';
import FileYourStateTaxesStepList from '../FileYourStateTaxesStepList/FileYourStateTaxesStepList.js';
import { REF_LOCATION, REF_LOCATION_VALUE } from '../../constants/pageConstants.js';

const FileYourStateTaxesDetails = () => {
  const { taxReturns } = useContext(TaxReturnsContext);
  const currentTaxReturn = getCurrentTaxYearReturn(taxReturns);
  const { stateProfile, isFetching } = useFetchStateProfile();

  const { t } = useTranslation();

  if (isFetching) {
    return <LoadingIndicator />;
  }

  if (stateProfile) {
    const context = {
      taxYear: currentTaxReturn?.taxYear,
      stateName: t(`enums.statesAndProvinces.${stateProfile.stateCode}`),
      stateTaxSystemName: stateProfile?.taxSystemName,
    };

    const buildLandingUrl = (urlString: string) => {
      const url = new URL(urlString);
      url.searchParams.append(REF_LOCATION, REF_LOCATION_VALUE.HOME);
      return url.toString();
    };

    const landingUrlWithRefLocationQueryParam = buildLandingUrl(stateProfile.landingUrl);

    return (
      <div>
        <h2>
          <Translation
            i18nKey='taxReturnCard.fileYourStateTaxesDetails.heading'
            collectionId={null}
            context={context}
          />
        </h2>
        {stateProfile.filingRequirementsUrl && ( // Only render this content if we have the link:
          <p>
            <Translation
              i18nKey='taxReturnCard.fileYourStateTaxesDetails.details'
              collectionId={null}
              context={context}
            />
            <span>
              &nbsp;
              <CommonLinkRenderer url={stateProfile.filingRequirementsUrl}>
                <Translation
                  i18nKey='taxReturnCard.fileYourStateTaxesDetails.learnAboutStateFilingRequirementsButtonText'
                  collectionId={null}
                  context={context}
                />
              </CommonLinkRenderer>
            </span>
          </p>
        )}
        <h3>
          <Translation
            i18nKey='taxReturnCard.fileYourStateTaxesDetails.subHeading'
            collectionId={null}
            context={context}
          />
        </h3>

        <FileYourStateTaxesStepList />

        <CommonLinkRenderer className='usa-button width-full margin-top-3' url={landingUrlWithRefLocationQueryParam}>
          <Translation
            i18nKey='taxReturnCard.fileYourStateTaxesDetails.startYourStateTaxesButtonText'
            collectionId={null}
            context={context}
          />
        </CommonLinkRenderer>
      </div>
    );
  }
};

export default FileYourStateTaxesDetails;
