import { Modal, ModalHeading, ModalRef } from '@trussworks/react-uswds';
import { Trans, useTranslation } from 'react-i18next';
import { RefObject } from 'react';

export type StatusInfoModalProps = {
  modalRef: RefObject<ModalRef>;
  canTransfer: boolean;
};

const StatusInfoModal = ({ modalRef, canTransfer }: StatusInfoModalProps) => {
  const { t } = useTranslation(`translation`, {
    keyPrefix: canTransfer ? `authorizeState.statusInfoModal` : `authorizeState.statusUpdateModal`,
  });

  return (
    <Modal
      id='status-info-modal'
      ref={modalRef}
      aria-labelledby='status-info-heading'
      aria-describedby='status-info-description'
    >
      <ModalHeading id='status-info-heading'>{t(`heading`)}</ModalHeading>
      <div className='usa-prose'>
        <p id='status-info-description'>
          <Trans t={t} i18nKey='content' />
        </p>
      </div>
    </Modal>
  );
};

export default StatusInfoModal;
