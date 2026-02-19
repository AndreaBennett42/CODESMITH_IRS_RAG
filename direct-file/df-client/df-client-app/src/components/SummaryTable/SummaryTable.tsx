import { SummaryBox, SummaryBoxContent, Table } from '@trussworks/react-uswds';
import { useTranslation } from 'react-i18next';
import styles from './SummaryTable.module.scss';
import { useConditionalListItems } from '../../hooks/useConditionalListItems.js';
import { ConditionalListProps, ItemConfig } from '../ConditionalList/ConditionalList.js';
import { useLinkComponents } from '@irs/df-common';
import { getModalOrTranslationComponent } from '../../utils/modalHelpers.js';
import { Link } from 'react-router-dom';
import Translation from '../Translation/index.js';
import classNames from 'classnames';

export type SummaryListItemConfig = ItemConfig & {
  internalLink?: string;
  showTopBorder?: boolean;
  indent?: boolean;
};

export type SummaryListProps = Omit<ConditionalListProps, `items`> & {
  items: SummaryListItemConfig[];
};

function withInternalLinkComponent(linkComponents: ReturnType<typeof useLinkComponents>, route: string | undefined) {
  if (route) {
    return {
      ...linkComponents,
      InternalLink: <Link to={route} />,
    };
  } else {
    return linkComponents;
  }
}

export function SummaryTable({ i18nKey, collectionId, items }: SummaryListProps) {
  const { i18n, t } = useTranslation();
  const actualKey = i18nKey.startsWith(`/info/`) ? `info.${i18nKey}` : i18nKey;

  const filterItems = useConditionalListItems(items, collectionId);
  const sharedLinkComponents = useLinkComponents(actualKey);
  const explainer = i18n.exists(`${actualKey}.sections.explainer`) ? (
    <p className='margin-0'>
      <Translation i18nKey={`${actualKey}.sections.explainer`} collectionId={null} />
    </p>
  ) : undefined;
  const conclusion = i18n.exists(`${actualKey}.sections.conclusion`) ? (
    <p className='margin-bottom-0'>
      <Translation i18nKey={`${actualKey}.sections.conclusion`} collectionId={null} />
    </p>
  ) : undefined;

  const visuallyHideIfExists = classNames({ 'usa-sr-only': i18n.exists(`${actualKey}.sections.caption`) });

  return (
    <SummaryBox className='margin-bottom-2'>
      <SummaryBoxContent>
        {explainer}
        <Table className={`${styles.table} margin-y-0`} fullWidth>
          <caption className={visuallyHideIfExists}>
            <Translation i18nKey={`${actualKey}.sections.caption`} collectionId={null} />
          </caption>
          <tbody>
            {filterItems.map(({ itemKey, showTopBorder, indent, internalLink, collectionId }) => {
              const sectionKey = `${actualKey}.sections.${itemKey}`;
              const baseValue = t(sectionKey, { returnObjects: true });
              const { Component, isModal } = getModalOrTranslationComponent(t, sectionKey);
              const { Component: THComponent } = getModalOrTranslationComponent(t, `${sectionKey}.th`);
              const { Component: TDComponent } = getModalOrTranslationComponent(t, `${sectionKey}.td`);
              const linkComponents = withInternalLinkComponent(sharedLinkComponents, internalLink);
              return (
                <tr key={`${itemKey}-${collectionId}`} className={showTopBorder ? styles.topBorder : ``}>
                  {!isModal && typeof baseValue === `object` ? (
                    <>
                      <th className={indent ? styles.indent : ``} scope='row'>
                        <THComponent
                          i18nKey={`${sectionKey}.th`}
                          collectionId={collectionId}
                          components={linkComponents}
                        />
                      </th>
                      <td className='text-right text-no-wrap'>
                        <TDComponent
                          i18nKey={`${sectionKey}.td`}
                          collectionId={collectionId}
                          components={linkComponents}
                        />
                      </td>
                    </>
                  ) : (
                    <td colSpan={2}>
                      <Component i18nKey={sectionKey} collectionId={collectionId} components={linkComponents} />
                    </td>
                  )}
                </tr>
              );
            })}
          </tbody>
        </Table>
        {conclusion}
      </SummaryBoxContent>
    </SummaryBox>
  );
}
