import { Modal, ModalHeading, ModalRef } from '@trussworks/react-uswds';
import { Trans, useTranslation } from 'react-i18next';
import { RefObject } from 'react';
import FileYourStateTaxesStepList from '../FileYourStateTaxesStepList/FileYourStateTaxesStepList.js';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import Translation from '../Translation/index.js';
import { StateProfile } from '../../types/StateProfile.js';

export type FileYourStateTaxesModalProps = {
  modalRef: RefObject<ModalRef>;
  stateProfile: StateProfile;
};

const FileYourStateTaxesModal = ({ modalRef, stateProfile }: FileYourStateTaxesModalProps) => {
  const { t } = useTranslation();

  const context = {
    stateName: t(`enums.statesAndProvinces.${stateProfile.stateCode}`),
    stateTaxSystemName: stateProfile?.taxSystemName,
  };

  return (
    <Modal
      id='status-info-modal'
      ref={modalRef}
      aria-labelledby='status-info-heading'
      aria-describedby='status-info-description'
    >
      <ModalHeading id='status-info-heading'>
        <Translation
          i18nKey='taxReturnCard.fileYourStateTaxesDetails.modalHeading'
          collectionId={null}
          context={context}
        />
      </ModalHeading>
      <div className='usa-prose'>
        {stateProfile.filingRequirementsUrl && ( // Only render this content if we have the link:
          <p id='status-info-description'>
            <Trans t={t} i18nKey='taxReturnCard.fileYourStateTaxesDetails.details' />
            {` `}
            <CommonLinkRenderer url={stateProfile.filingRequirementsUrl}>
              <Translation
                i18nKey='taxReturnCard.fileYourStateTaxesDetails.learnAboutStateFilingRequirementsButtonText'
                collectionId={null}
                context={context}
              />
            </CommonLinkRenderer>
          </p>
        )}
        <h3>
          <Translation
            i18nKey='taxReturnCard.fileYourStateTaxesDetails.subheading'
            collectionId={null}
            context={context}
          />
        </h3>

        <div className='margin-top-2'>
          <FileYourStateTaxesStepList />
        </div>

        {stateProfile.departmentOfRevenueUrl && ( // Only render this content if we have the link:
          <p>
            <Trans t={t} i18nKey='taxReturnCard.fileYourStateTaxesDetails.modalFooter' />
            {` `}
            <CommonLinkRenderer url={stateProfile.departmentOfRevenueUrl}>
              <Translation
                i18nKey='taxReturnCard.fileYourStateTaxesDetails.stateDorSite'
                collectionId={null}
                context={context}
              />
            </CommonLinkRenderer>
          </p>
        )}
      </div>
    </Modal>
  );
};

export default FileYourStateTaxesModal;
