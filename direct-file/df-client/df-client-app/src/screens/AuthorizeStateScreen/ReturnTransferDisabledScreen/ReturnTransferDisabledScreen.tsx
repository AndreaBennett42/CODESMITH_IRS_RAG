import { Button, GridContainer, ModalRef } from '@trussworks/react-uswds';
import PageTitle from '../../../components/PageTitle/index.js';
import { Trans, useTranslation } from 'react-i18next';
import useTranslatePIIRedacted from '../../../hooks/useTranslatePIIRedacted.js';
import { MouseEventHandler, useRef } from 'react';
// eslint-disable-next-line max-len
import ReturnTransferDisabledMoreInfoModal from './ReturnTransferDisabledMoreInfoModal/ReturnTransferDisabledMoreInfoModal.js';
import classNames from 'classnames';
import { useFetchPdf } from '../../../hooks/useApiHook.js';

type ReturnTransferDisabledScreenProps = {
  taxReturnUuid: string;
};
const ReturnTransferDisabledScreen = ({ taxReturnUuid }: ReturnTransferDisabledScreenProps) => {
  const { t } = useTranslation(`translation`, { keyPrefix: `authorizeState.transferDisabled` });
  const redacted = useTranslatePIIRedacted(`authorizeState.transferDisabled.heading`, true, {});
  const { loading, fetchPdf } = useFetchPdf();

  const moreInfoModalRef = useRef<ModalRef>(null);

  const handleClickDownloadPdfButton: MouseEventHandler<HTMLButtonElement> = async (_e) => {
    if (!loading) {
      void fetchPdf(taxReturnUuid);
    }
  };

  const handleClickMoreInfoButtonText: MouseEventHandler<HTMLButtonElement> = (e) => {
    moreInfoModalRef.current?.toggleModal(e, true);
  };

  const buttonClasses = classNames({
    'usa-button--disabled': loading,
  });

  return (
    <>
      <ReturnTransferDisabledMoreInfoModal modalRef={moreInfoModalRef} />
      <GridContainer>
        <PageTitle redactedTitle={redacted}>{t(`heading`)}</PageTitle>
        <div className='usa-prose margin-top-3'>
          <Trans
            t={t}
            i18nKey='content'
            components={{
              downloadPdfButton: (
                <Button type='button' className={buttonClasses} unstyled onClick={handleClickDownloadPdfButton}>
                  {t(`downloadPdfButtonText`)}
                </Button>
              ),
            }}
          >
            content
          </Trans>
        </div>
        <Button unstyled type='button' className='margin-top-2' onClick={handleClickMoreInfoButtonText}>
          {t(`moreInfoButtonText`)}
        </Button>
      </GridContainer>
    </>
  );
};

export default ReturnTransferDisabledScreen;
