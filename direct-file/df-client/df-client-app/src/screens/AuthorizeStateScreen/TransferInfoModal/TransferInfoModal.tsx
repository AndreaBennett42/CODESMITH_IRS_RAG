import { Button, Modal, ModalHeading, ModalRef } from '@trussworks/react-uswds';
import { Trans, useTranslation } from 'react-i18next';
import { MouseEventHandler, RefObject } from 'react';
import { useFetchPdf } from '../../../hooks/useApiHook.js';
import classNames from 'classnames';
import TransferableDataList from './TransferredDataList/TransferableDataList.js';

export type TransferInfoModalProps = {
  modalRef: RefObject<ModalRef>;
  taxReturnUuid: string;
};

/**
 * Modal that shows the user what information is transferred to states, thereby disclosing it to the taxpayer
 * who then opts to transfer this information as non-FTI to the state.
 */
const TransferInfoModal = ({ modalRef, taxReturnUuid }: TransferInfoModalProps) => {
  const { t } = useTranslation(`translation`, { keyPrefix: `authorizeState.transferReturn.transferInfoModal` });
  const { loading, fetchPdf } = useFetchPdf();
  const handleClickDownloadPdfButton: MouseEventHandler<HTMLButtonElement> = async (_e) => {
    if (!loading) {
      void fetchPdf(taxReturnUuid);
    }
  };

  const buttonClasses = classNames({
    'usa-button--disabled': loading,
  });

  return (
    <Modal
      id='transfer-info-modal'
      ref={modalRef}
      aria-labelledby='transfer-info-heading'
      aria-describedby='transfer-info-description'
    >
      <ModalHeading id='transfer-info-heading'>{t(`heading`)}</ModalHeading>
      <div className='usa-prose'>
        <p id='transfer-info-description'>
          <Trans t={t} i18nKey='content' />
        </p>
      </div>
      <Button type='button' className={buttonClasses} unstyled onClick={handleClickDownloadPdfButton}>
        {t(`downloadPdfButtonText`)}
      </Button>
      <TransferableDataList />
    </Modal>
  );
};

export default TransferInfoModal;
