import { useContext, useRef, useState } from 'react';
import { Alert, Button, ModalRef } from '@trussworks/react-uswds';
import { useTranslation } from 'react-i18next';
import SurveyModal from './SurveyModal/SurveryModal.js';
import { TaxReturnsContext } from '../context/TaxReturnsContext.js';
import { getTaxReturnById } from '../utils/taxReturnUtils.js';
import { useFactGraph } from '../factgraph/FactGraphContext.js';
import { Path } from '../flow/Path.js';
import SurveryConfirmationModal from './SurveyConfirmationModal.js';

const SurveyBanner = () => {
  const surveyModalRef = useRef<ModalRef>(null);
  const confirmationModalRef = useRef<ModalRef>(null);
  const { t } = useTranslation(`translation`);
  const modalButtonText = t(`surveyBanner.header`);
  const [showSurveyModal, setShowSurveyModal] = useState(false);
  const [showConfirmationModal, setShowConfirmationModal] = useState(false);
  const [confirmationType, setConfirmationType] = useState<'yes' | 'no-thanks' | null>(null);
  const { currentTaxReturnId, taxReturns } = useContext(TaxReturnsContext);
  const currentTaxReturn = getTaxReturnById(taxReturns, currentTaxReturnId);
  const { factGraph } = useFactGraph();
  const isResubmittingFactResult = factGraph.get(Path.concretePath(`/isResubmitting`, null));

  const openSurveyModal: React.MouseEventHandler<HTMLButtonElement> = (e) => {
    setShowSurveyModal(true);
    setShowConfirmationModal(false);
    setConfirmationType(null);
    if (surveyModalRef.current && !surveyModalRef.current.modalIsOpen) {
      surveyModalRef.current.toggleModal(e, true); // Ensure it opens
    }
  };

  const openSurveyConfirmationModal = (type: 'yes' | 'no-thanks') => {
    setConfirmationType(type);
    setShowConfirmationModal(true);
    setShowConfirmationModal(true);
    if (confirmationModalRef.current) {
      const fakeEvent = new Event(`click`, { bubbles: true }) as unknown as React.MouseEvent<HTMLButtonElement>;
      confirmationModalRef.current.toggleModal(fakeEvent, true); // Force open
    }
  };

  const toggleSurveyModal = (e: React.MouseEvent<HTMLButtonElement>) => {
    setShowSurveyModal(false);
    if (surveyModalRef.current && surveyModalRef.current.modalIsOpen) {
      surveyModalRef.current.toggleModal(e, false); // Ensure modal closes
    }
  };

  const toggleConfirmationModal = () => {
    setShowConfirmationModal(false);
    setConfirmationType(null);
  };

  if ((currentTaxReturn?.surveyOptIn !== null && !showConfirmationModal) || isResubmittingFactResult.complete) {
    return null;
  }

  return (
    <section aria-label={t(`banner.connectivity.name`)}>
      {showSurveyModal && (
        <SurveyModal
          modalRef={surveyModalRef}
          toggleModal={toggleSurveyModal}
          openSurveyConfirmationModal={openSurveyConfirmationModal}
        />
      )}
      {showConfirmationModal && confirmationType && (
        <>
          <SurveryConfirmationModal
            confirmationType={confirmationType}
            modalRef={confirmationModalRef}
            toggleModal={toggleConfirmationModal}
          />
        </>
      )}
      <Alert type={`info`} headingLevel='h6' role='alert'>
        <Button type='button' unstyled onClick={openSurveyModal}>
          {modalButtonText}
        </Button>
      </Alert>
    </section>
  );
};

export default SurveyBanner;
