import { Modal, ModalHeading, ModalRef, Icon } from '@trussworks/react-uswds';
import Translation from '../components/Translation/Translation.js';
import { useEffect, useRef } from 'react';

type ConfirmationType = 'yes' | 'no-thanks';

interface ConfirmationModalProps {
  modalRef: React.RefObject<ModalRef>;
  confirmationType: ConfirmationType;
  toggleModal: (e: React.MouseEvent<HTMLButtonElement>) => void;
}

const SurveryConfirmationModal = ({ confirmationType, modalRef, toggleModal }: ConfirmationModalProps) => {
  const headerTextKey = confirmationType === `yes` ? `surveyBanner.shareEmailHeading` : `surveyBanner.noThanksHeading`;
  const internalModalRef = useRef<ModalRef>(null);
  const refToUse = modalRef || internalModalRef;

  useEffect(() => {
    if (refToUse.current && !refToUse.current.modalIsOpen) {
      const fakeEvent = new Event(`click`, { bubbles: true }) as unknown as React.MouseEvent<HTMLButtonElement>;
      refToUse.current.toggleModal(fakeEvent, true); // Ensure it opens
    }

    // Sync state when "X" closes the modal
    const checkClose = () => {
      if (refToUse.current && !refToUse.current.modalIsOpen) {
        const fakeEvent = new Event(`click`, { bubbles: true }) as unknown as React.MouseEvent<HTMLButtonElement>;
        toggleModal(fakeEvent); // Sync state with SurveyBanner
      }
    };

    const interval = setInterval(checkClose, 100); // Check every 100ms
    return () => {
      clearInterval(interval);
    };
  }, [refToUse, toggleModal]);

  return (
    <Modal
      ref={refToUse}
      id='confirmation-modal'
      aria-labelledby='confirmation-modal-heading'
      aria-describedby='confirmation-modal-description'
      isInitiallyOpen={true}
    >
      {confirmationType === `yes` && (
        <div style={{ textAlign: `center`, marginBottom: `1rem` }}>
          <Icon.CheckCircle size={9} color='green' />
        </div>
      )}
      <ModalHeading id='confirmation-modal-heading'>
        <Translation i18nKey={headerTextKey} collectionId={null} />
      </ModalHeading>
      {confirmationType === `yes` && (
        <div id='confirmation-modal-description'>
          <p>
            <Translation i18nKey={`surveyBanner.shareEmailText`} collectionId={null} />
          </p>
        </div>
      )}
    </Modal>
  );
};

export default SurveryConfirmationModal;
