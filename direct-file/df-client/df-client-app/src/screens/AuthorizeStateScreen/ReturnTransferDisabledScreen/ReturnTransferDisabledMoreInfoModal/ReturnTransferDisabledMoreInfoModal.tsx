import { RefObject } from 'react';
import { Modal, ModalHeading, ModalRef } from '@trussworks/react-uswds';
import { Trans, useTranslation } from 'react-i18next';

export type ReturnTransferDisabledMoreInfoModalProps = {
  modalRef: RefObject<ModalRef>;
};
const ReturnTransferDisabledMoreInfoModal = ({ modalRef }: ReturnTransferDisabledMoreInfoModalProps) => {
  const { t } = useTranslation(`translation`, {
    keyPrefix: `authorizeState.transferDisabled.transferDisabledMoreInfoModal`,
  });

  return (
    <Modal
      id='return-transfer-disabled-more-info-modal'
      ref={modalRef}
      aria-labelledby='return-transfer-disabled-more-info-heading'
      aria-describedby='return-transfer-disabled-more-info-description'
    >
      <ModalHeading id='return-transfer-disabled-more-info-heading'>{t(`heading`)}</ModalHeading>
      <div className='usa-prose'>
        <p id='return-transfer-disabled-more-info-description'>
          <Trans t={t} i18nKey='content' />
        </p>
      </div>
    </Modal>
  );
};

export default ReturnTransferDisabledMoreInfoModal;
