import { createRef } from 'react';
import { Modal, ModalRef, ModalHeading } from '@trussworks/react-uswds';
import Translation from '../Translation/index.js';
import { useTranslation } from 'react-i18next';
import { CommonTranslation } from 'df-i18n';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import { InfoDisplayProps } from '../../types/core.js';
import { v4 as uuidv4 } from 'uuid';
import ContentDisplay, { BareContentDisplay } from '../ContentDisplay/index.js';
import { ItemConfig } from '../ConditionalList/ConditionalList.js';
import { useConditionalListItems } from '../../hooks/useConditionalListItems.js';
import { calculateScreenStatus } from '../../flow/batches.js';
import classNames from 'classnames';
import screenStyles from '../Screen.module.scss';

export type DFModalProps = Omit<InfoDisplayProps, 'gotoNextScreen' | 'i18nKey' | 'collectionId'> & {
  i18nKey: string | string[];
  additionalTextComponents?: Record<string, JSX.Element>;
  items?: ItemConfig[];
  collectionId: string | null;
  additionalComponents?: Record<string, JSX.Element>;
  context?: object | undefined;
};

type ModalInfo = {
  key: string;
  i18nKey: string;
  ref: React.RefObject<ModalRef>;
  linkComponent: JSX.Element;
};

export type HamlTree =
  | string
  | Array<{
      [key: string]: HamlTree;
    }>;

// Extracts all tags from a haml tree in the yaml content
export const extractTags = (content: HamlTree) => {
  const tags: string[] = [];
  const extractRecurse = (content: HamlTree): string[] => {
    if (Array.isArray(content)) {
      return content.flatMap((item) => extractRecurse(Object.values(item)[0]));
    } else if (typeof content === `string`) {
      return content.match(/<[^/>]+>/g) ?? [];
    }
    return tags;
  };
  return extractRecurse(content);
};

const DFModal = ({
  i18nKey,
  collectionId = null,
  additionalTextComponents,
  additionalComponents,
  items,
  context,
  batches,
}: DFModalProps) => {
  const { t, i18n } = useTranslation();
  const i18nKeys = Array.isArray(i18nKey) ? i18nKey : [i18nKey];
  const fullKey = CommonTranslation.getFallbackKey(i18nKeys, i18n);
  const isOpen = calculateScreenStatus(batches).isOpen;
  const conditionalItems = useConditionalListItems(items || [], collectionId);
  const filteredItems: ItemConfig[] = items ? conditionalItems : [];

  // This is because the FormFieldWrapper calls this component with a key that doesn't exist
  // to manage edge case (email) where the hint is a DFModal
  // TODO should be fixed in the FormFieldWrapper
  // https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/11795
  if (!fullKey || !i18n.exists(`${fullKey}.helpText.modals`)) {
    return null;
  }

  const filteredItemKeys = new Set(filteredItems.map((item) => item.itemKey));
  // If we're filtering items, and none of them are true, don't render anything
  if (items && filteredItemKeys.size === 0) {
    return null;
  }

  const modalInfos: ModalInfo[] = [];

  const modals: Record<string, unknown> = t(`${fullKey}.helpText.modals`, { returnObjects: true }) as Record<
    string,
    unknown
  >;

  // Get all modal tags from text
  // - LinkModals will be kept with this instance
  // - sharedModals will be kept under modals (and used in other pages)
  //   so must have unique keys

  const modalTags = extractTags(modals.text as HamlTree).filter(
    (tag) => tag.startsWith(`<LinkModal`) || tag.startsWith(`<sharedModal`)
  );
  const modalKeys = modalTags ? modalTags.map((tag) => tag.replace(/<|>/g, ``)) : [];

  // Collect all modal keys and content, either from LinkModal or sharedModal
  modalKeys
    .filter((key) => key.startsWith(`LinkModal`) || key.startsWith(`sharedModal`))
    .forEach((key) => {
      const ref = createRef<ModalRef>();
      const i18nKey = key.startsWith(`LinkModal`) ? `${fullKey}.helpText.modals.${key}` : `modals.${key}`;
      modalInfos.push({
        key,
        i18nKey,
        ref,
        linkComponent: <CommonLinkRenderer url='' modalRef={ref} />,
      });
    });

  // Create an object with the link components
  const linkComponents: Record<string, JSX.Element> = {};
  modalInfos.forEach((modalInfo) => {
    linkComponents[modalInfo.key] = modalInfo.linkComponent;
  });

  // guess whether the launcher text is near a form field element
  const calculatedClass = fullKey.includes(`info`) ? `margin-y-1` : ``;

  // TODO: Eliminate the need for this
  const suppressMargin = fullKey.includes(`rejectedReturnDetails`);

  /** This is the text displayed on the screen, with the Link to launch the modal. */
  const snackText = (
    <BareContentDisplay
      collectionId={collectionId}
      i18nKey={`${fullKey}.helpText.modals`}
      additionalComponents={{
        ...linkComponents,
        ...additionalTextComponents,
      }}
      subKey='text'
    />
  );

  const allowedTags = [`h2`, `p`, `ul`, `li`, `a`, `ol`];
  return (
    <>
      {suppressMargin ? snackText : <div className={calculatedClass}>{snackText}</div>}
      {modalInfos.map(({ key: modalKey, i18nKey: modalContenti18nKey, ref: modalRef }) => {
        const modalData: { [index: string]: unknown } = t(modalContenti18nKey, {
          returnObjects: true,
        }) as { [index: string]: unknown };
        let content;

        /* content has a `body` key, so if the top level doesn't have `body` then it's a conditional key */
        if (items && !modalData.body) {
          const filteredModalData: Record<string, typeof modalData> = {};
          for (const key of Object.keys(modalData)) {
            if (filteredItemKeys.has(key)) {
              filteredModalData[key] = modalData[key] as Record<string, typeof modalData>;
            }
          }

          // handle the case where filters are present but all false; don't try to show a modal with no content
          if (Object.keys(filteredModalData).length === 0) {
            return null;
          }

          // if we have conditional keys, iterate over them and render the content
          content = Object.keys(filteredModalData).map((key) => (
            <ContentDisplay
              key={key}
              context={context}
              collectionId={collectionId}
              i18nKey={`${modalContenti18nKey}.${key}`}
              additionalComponents={{
                h2: <h2 className='usa-modal__heading' />,
                ...additionalComponents,
              }}
              allowedTags={allowedTags}
            />
          ));
        } else {
          // If "body" is in the top level of modalData, we do our regular rendering
          content = (
            <ContentDisplay
              context={context}
              collectionId={collectionId}
              i18nKey={modalContenti18nKey}
              additionalComponents={{
                h2: <h2 className='usa-modal__heading' />,
                ...additionalComponents,
              }}
              allowedTags={allowedTags}
            />
          );
        }

        const uniqueModalKey = `${fullKey}-${modalKey}`;
        const uniqueModalId = uuidv4();

        return (
          <Modal
            key={uniqueModalKey}
            ref={modalRef}
            id={uniqueModalKey}
            aria-labelledby={`modal-heading-${uniqueModalId}`}
            aria-describedby={`modal-description-${uniqueModalId}`}
          >
            <ModalHeading
              id={`modal-heading-${uniqueModalId}`}
              className={classNames({ [screenStyles.draftContent]: isOpen })}
            >
              <Translation collectionId={collectionId} i18nKey={`${modalContenti18nKey}.header`} />
            </ModalHeading>
            <div
              className={classNames(`usa-prose`, { [screenStyles.draftContent]: isOpen })}
              id={`modal-description-${uniqueModalId}`}
            >
              {content}
            </div>
          </Modal>
        );
      })}
    </>
  );
};

export default DFModal;
