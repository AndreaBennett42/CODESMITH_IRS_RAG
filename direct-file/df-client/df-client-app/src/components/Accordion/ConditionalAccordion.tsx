import { FC, Fragment, useState } from 'react';
import { Accordion } from '@trussworks/react-uswds';
import { ConditionalList, ConditionalListProps, ItemConfig } from '../ConditionalList/ConditionalList.js';
import { InfoDisplayProps } from '../../types/core.js';
import type { CommonAccordionItemProps } from '@irs/df-common';
import Translation from '../Translation/index.js';
import styles from './ConditionalAccordion.module.scss';
import { CommonTranslation } from 'df-i18n';
import { useTranslation } from 'react-i18next';
import { useConditionalListItems } from '../../hooks/useConditionalListItems.js';

export type ConditionalAccordionProps = InfoDisplayProps & ConditionalListProps & AdditionalAccordionProps;

type ConditionalSubListProps = {
  subListKey: string;
  collectionId: string | null;
  items: ItemConfig[];
  i18nKey: string;
  displayOnlyHeaders: boolean;
};

type AdditionalAccordionProps = {
  displayOnlyHeaders?: boolean;
};

const ConditionalSubList: FC<ConditionalSubListProps> = ({
  subListKey,
  collectionId,
  items,
  i18nKey,
  displayOnlyHeaders = false,
}) => {
  const subListItems = displayOnlyHeaders
    ? // eslint-disable-next-line eqeqeq
      items.filter((item) => item.itemKey == subListKey)
    : items.filter((item) => item.itemKey.startsWith(`${subListKey}-`));
  const conditionalItems = useConditionalListItems(subListItems, collectionId);
  if (conditionalItems.length === 0) {
    return null;
  }

  if (displayOnlyHeaders) {
    return (
      <p key={subListKey}>
        <Translation i18nKey={`${i18nKey}.${subListKey}`} collectionId={collectionId} />
      </p>
    );
  } else {
    return (
      <Fragment key={subListKey}>
        <Translation i18nKey={`${i18nKey}.${subListKey}`} collectionId={collectionId} />
        <ConditionalList items={conditionalItems} collectionId={collectionId} i18nKey={i18nKey} />
      </Fragment>
    );
  }
};

const ConditionalAccordion: FC<ConditionalAccordionProps> = (props) => {
  const [expanded, setExpanded] = useState(false);
  const { t, i18n } = useTranslation();
  const nameSpacedKey = CommonTranslation.getNamespacedKey(props.i18nKey);
  const displayOnlyHeaders = props.displayOnlyHeaders ?? false;
  if (!i18n.exists(nameSpacedKey)) {
    return null;
  }
  const objectAtYaml = t(nameSpacedKey, { returnObjects: true });
  const subListKeys = Object.keys(objectAtYaml).filter((key) => key.startsWith(`subList`));
  const subListKeysWithoutHypens = subListKeys.filter((key) => key.indexOf(`-`) === -1);

  const items: CommonAccordionItemProps[] = [
    {
      id: props.i18nKey,
      title: <Translation i18nKey={`${nameSpacedKey}.heading`} collectionId={props.collectionId} />,
      content: (
        <>
          {subListKeysWithoutHypens.map((subListKey) => (
            <ConditionalSubList
              key={subListKey}
              subListKey={subListKey}
              collectionId={props.collectionId}
              items={props.items}
              i18nKey={nameSpacedKey}
              displayOnlyHeaders={displayOnlyHeaders}
            />
          ))}
        </>
      ),
      expanded,
      headingLevel: `h2`,
      handleToggle: () => setExpanded((previousValue) => !previousValue),
    },
  ];

  return (
    <>
      <Accordion bordered items={items} className={styles.accordion} />
    </>
  );
};

export default ConditionalAccordion;
