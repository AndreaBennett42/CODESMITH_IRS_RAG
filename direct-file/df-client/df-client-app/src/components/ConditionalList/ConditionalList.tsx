import { useTranslation } from 'react-i18next';
import { RawCondition } from '../../flow/Condition.js';
import { CommonTranslation } from 'df-i18n';
import { getModalOrTranslationComponent } from '../../utils/modalHelpers.js';
import { useConditionalListItems } from '../../hooks/useConditionalListItems.js';
import { SummaryBox, SummaryBoxContent } from '@trussworks/react-uswds';
import { FC } from 'react';
import { AbsolutePath } from '../../fact-dictionary/Path.js';
import ContentDisplay from '../ContentDisplay/index.js';

export type ItemConfig = {
  editRoute?: string;
  itemKey: string;
  conditions?: RawCondition[];
  /** If a collection is provided, will repeat the item for each member in a collection */
  collection?: AbsolutePath;
};

export type ConditionalListProps = {
  i18nKey: string;
  items: ItemConfig[];
  collectionId: string | null;
  isSummary?: boolean;
  i18nPrefixKey?: string;
};

export const ConditionalList: FC<ConditionalListProps> = ({
  i18nKey,
  items,
  collectionId,
  isSummary = false,
  i18nPrefixKey,
}: ConditionalListProps) => {
  const { t, i18n } = useTranslation();
  const nameSpacedKey = CommonTranslation.getNamespacedKey(i18nKey);
  const nameSpacedPrefixKey = i18nPrefixKey && CommonTranslation.getNamespacedKey(i18nPrefixKey);
  const filteredItems = useConditionalListItems(items, collectionId);

  const UnorderedList = () => {
    return (
      <>
        {nameSpacedPrefixKey && <ContentDisplay i18nKey={nameSpacedPrefixKey} />}
        <ul>
          {filteredItems.map((item) => {
            const fullI18nKey = `${nameSpacedKey}.${item.itemKey}`;
            if (!i18n.exists(fullI18nKey)) {
              return null;
            }
            const { Component, hasInternalLink, InternalLink } = getModalOrTranslationComponent(t, fullI18nKey);
            return (
              <li key={`${fullI18nKey}-${item.collectionId}-${item.conditions?.join(`,`)}`} className='screen__info'>
                {hasInternalLink && item.editRoute ? (
                  <InternalLink i18nKey={fullI18nKey} collectionId={item.collectionId} route={item.editRoute} />
                ) : (
                  <Component i18nKey={fullI18nKey} collectionId={item.collectionId} />
                )}
              </li>
            );
          })}
        </ul>
      </>
    );
  };

  return isSummary ? (
    <SummaryBox data-testid='conditional-list-summary-box'>
      <SummaryBoxContent>
        <UnorderedList />
      </SummaryBoxContent>
    </SummaryBox>
  ) : (
    <UnorderedList />
  );
};
