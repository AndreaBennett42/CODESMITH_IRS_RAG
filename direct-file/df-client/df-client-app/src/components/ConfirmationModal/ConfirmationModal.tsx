import { createRef } from 'react';
import {
  Modal,
  ModalRef,
  ModalHeading,
  ModalFooter,
  ButtonGroup,
  ModalToggleButton,
  Icon,
} from '@trussworks/react-uswds';
import Translation from '../Translation/index.js';
import { useTranslation } from 'react-i18next';
import { CommonTranslation } from 'df-i18n';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import { InfoDisplayProps } from '../../types/core.js';
import { v4 as uuidv4 } from 'uuid';
import ContentDisplay, { BareContentDisplay } from '../ContentDisplay/index.js';

export type ConfirmationModalProps = Omit<InfoDisplayProps, 'gotoNextScreen' | `i18nKey`> & {
  i18nKey: string;
  handleConfirm: () => void;
  handleCancel?: () => void;
  additionalTextComponents?: Record<string, JSX.Element>;
  destructiveAction?: boolean;
  modalOpenerClasses?: string;
  icon?: Exclude<keyof typeof Icon, 'prototype'>;
};

const ConfirmationModal = ({
  i18nKey,
  collectionId,
  handleConfirm,
  handleCancel,
  additionalTextComponents,
  destructiveAction,
  modalOpenerClasses,
  icon,
}: ConfirmationModalProps) => {
  const { t, i18n } = useTranslation();
  const fullKey = CommonTranslation.getNamespacedKey(i18nKey);
  if (!i18n.exists(fullKey)) {
    return null;
  }

  const modals: Record<string, unknown> = t(fullKey, { returnObjects: true }) as Record<string, unknown>;

  const modalKey = Object.keys(modals).find((key) => key.startsWith(`LinkModal`));
  const modalRef = createRef<ModalRef>();

  // prep link element that will be used to launch the modal
  const linkComponents: Record<string, JSX.Element> = {
    LinkModal1: <CommonLinkRenderer url='' modalRef={modalRef} modalOpenerClasses={modalOpenerClasses} icon={icon} />,
  };

  // guess whether the launcher text is near a form field element
  const calculatedClass = fullKey.includes(`info`) ? `margin-y-1` : ``;

  const uniqueModalKey = `${fullKey}-${modalKey}`;
  const uniqueModalId = uuidv4();

  // guess whether the launcher text is inside of an alert
  const isInAlert = fullKey.includes(`alertText`);

  /** This is the text displayed on the screen, with the Link to launch the modal. */
  const snackText = (
    <BareContentDisplay
      collectionId={collectionId}
      i18nKey={fullKey}
      additionalComponents={{
        ...linkComponents,
        ...additionalTextComponents,
      }}
      subKey='text'
    />
  );

  return (
    <>
      {isInAlert ? snackText : <div className={calculatedClass}>{snackText}</div>}
      <Modal
        key={uniqueModalKey}
        ref={modalRef}
        id={uniqueModalKey}
        forceAction
        aria-labelledby={`modal-heading-${uniqueModalId}`}
        aria-describedby={`modal-description-${uniqueModalId}`}
      >
        <ModalHeading id={`modal-heading-${uniqueModalId}`}>
          <Translation collectionId={collectionId} i18nKey={`${fullKey}.${modalKey}.header`} />
        </ModalHeading>
        <div className='usa-prose' id={`modal-description-${uniqueModalId}`}>
          <ContentDisplay
            collectionId={collectionId}
            i18nKey={`${fullKey}.${modalKey}`}
            additionalComponents={{
              h2: <h2 className='usa-modal__heading' />,
            }}
          />
        </div>
        <ModalFooter>
          <ButtonGroup>
            <ModalToggleButton modalRef={modalRef} onClick={handleConfirm} closer secondary={destructiveAction}>
              <Translation collectionId={collectionId} i18nKey={`${fullKey}.${modalKey}.controls.confirm`} />
            </ModalToggleButton>
            <ModalToggleButton modalRef={modalRef} onClick={handleCancel} closer unstyled>
              <Translation collectionId={collectionId} i18nKey={`${fullKey}.${modalKey}.controls.cancel`} />
            </ModalToggleButton>
          </ButtonGroup>
        </ModalFooter>
      </Modal>
    </>
  );
};

export default ConfirmationModal;
