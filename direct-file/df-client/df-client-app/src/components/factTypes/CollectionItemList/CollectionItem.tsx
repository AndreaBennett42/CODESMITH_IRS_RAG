import { Alert, Card, CardBody, CardFooter, CardHeader } from '@trussworks/react-uswds';
import IconDisplay, { IconName } from '../../IconDisplay/IconDisplay.js';
import styles from './CollectionItem.module.scss';
import Translation from '../../Translation/index.js';
import { Link } from 'react-router-dom';
import { findFirstIncompleteScreenOfLoop } from '../../../flow/flowHelpers.js';
import { useFactGraph } from '../../../factgraph/FactGraphContext.js';
import { FlowCollectionLoop } from '../../../flow/flowConfig.js';
import { AlertConfigs, MefAlertConfig } from '../../../misc/aggregatedAlertHelpers.js';
import Assertion from '../../../screens/data-view/Assertion.js';
import { AssertionDeclaration } from '../../../flow/ContentDeclarations.js';
import useTranslateWithFacts from '../../../hooks/useTranslateWithFacts.js';
import { useTranslation } from 'react-i18next';
import { renderMefAlertForDataView, renderTaxReturnAlertForDataView } from '../../../screens/DataView.js';
import { handleRefFromRoute } from '../../SummaryAlert/summaryHelpers.js';
import { RefObject } from 'react';
import { useKnockoutCheck } from '../../../hooks/useKnockoutCheck.js';

interface CollectionItemProps {
  itemPosition: number;
  iconName?: IconName;
  detailLink: string;
  itemId: string;
  path: string;
  inProgress: boolean;
  loop: FlowCollectionLoop;
  alertConfigs?: AlertConfigs;
  mefAlertConfigs?: AlertConfigs<MefAlertConfig>;
  cardHeadingLevel: number;
  hideCardLabel2?: boolean;
  assertion?: AssertionDeclaration;
  refs: React.MutableRefObject<Map<string, React.MutableRefObject<HTMLHeadingElement>>>;
  aggregateAlertKey: string | undefined;
}

const CollectionItem = ({
  itemPosition,
  iconName,
  detailLink,
  itemId,
  path,
  inProgress,
  loop,
  alertConfigs,
  mefAlertConfigs,
  cardHeadingLevel,
  hideCardLabel2,
  assertion,
  refs,
  aggregateAlertKey,
}: CollectionItemProps) => {
  const { factGraph } = useFactGraph();
  const { i18n, t } = useTranslation();
  const { contextHasData } = useTranslateWithFacts(factGraph, itemId);
  const { getIsKnockedOut } = useKnockoutCheck();
  const isKnockedOut = getIsKnockedOut();

  const nextIncompleteScreen = inProgress && findFirstIncompleteScreenOfLoop(loop, factGraph, itemId);
  const Heading = cardHeadingLevel === 2 ? `h2` : `h3`;

  const i18nKeys = (leafKey: string) => {
    return [
      `fields.${loop.loopName}.collectionListing.${leafKey}`,
      `fields.${loop.collectionName}.collectionListing.${leafKey}`,
    ];
  };
  const primaryItemHeadingKey = i18nKeys(`itemHeading1`);
  const secondaryItemHeadingKey = i18nKeys(`itemHeading2`);

  const hasPrimaryItemHeading = i18n.exists(primaryItemHeadingKey) && contextHasData(primaryItemHeadingKey);
  const primaryItemLabel1 = i18nKeys(`label1`);
  const hasPrimaryItemLabel1 = primaryItemLabel1.some((key) => i18n.exists(key));
  const primaryItemValue1 = i18nKeys(`value1`);
  const hasPrimaryItemValue1 = primaryItemValue1.some((key) => i18n.exists(key));
  const primaryItemValue2 = i18nKeys(`value2`);
  const hasPrimaryItemValue2 = primaryItemValue2.some((key) => i18n.exists(key));
  const headingKey = hasPrimaryItemHeading ? primaryItemHeadingKey : secondaryItemHeadingKey;
  const collectionItemType =
    path === `/familyAndHousehold` ? t(`accessibility.screenReaderPerson`) : t(`accessibility.screenReaderIncome`);
  const ScreenReaderText = () => <span className='usa-sr-only'>{`${collectionItemType} #${itemPosition}`}</span>;
  return (
    <Card key={itemId} className={styles.dfCollectionItem} gridLayout={{ tablet: { col: 12 } }}>
      <CardHeader>
        <div className={styles.headerRow}>
          {iconName && <IconDisplay name={iconName} size={3} className={styles.icon} />}
          <Heading
            tabIndex={-1}
            className='usa-card__heading'
            ref={
              aggregateAlertKey
                ? (handleRefFromRoute(aggregateAlertKey, refs) as RefObject<HTMLHeadingElement>)
                : undefined
            }
          >
            <ScreenReaderText />
            <Translation i18nKey={headingKey} collectionId={itemId} />
          </Heading>
        </div>
      </CardHeader>

      <CardBody>
        <div className={styles.bodyRow}>
          {iconName && <span className={styles.spacer}></span>}
          <div>
            <ul>
              {hasPrimaryItemHeading && hasPrimaryItemLabel1 && (
                <li>
                  <Translation i18nKey={primaryItemLabel1} collectionId={itemId} />
                  <br />
                  {hasPrimaryItemValue1 && <Translation i18nKey={primaryItemValue1} collectionId={itemId} />}
                </li>
              )}
              {!inProgress && !hideCardLabel2 && (
                <li data-testid='cardLabel2'>
                  <Translation i18nKey={i18nKeys(`label2`)} collectionId={itemId} />
                  <br />
                  {hasPrimaryItemValue2 && <Translation i18nKey={primaryItemValue2} collectionId={itemId} />}
                </li>
              )}
            </ul>
            {inProgress && !isKnockedOut && nextIncompleteScreen && (
              <Alert className=' margin-top-3' type='error' headingLevel='h3' validation slim>
                <Translation i18nKey='dataviews.incompleteSection' collectionId={null} />
                &nbsp;
                <Link to={detailLink}>
                  <Translation i18nKey='dataviews.review' collectionId={null} />
                </Link>
              </Alert>
            )}
            {assertion && assertion.i18nKey && (
              <Assertion type={assertion.type} i18nKey={assertion.i18nKey} collectionId={itemId} />
            )}
            {mefAlertConfigs?.errors && mefAlertConfigs.errors.map((mefError) => renderMefAlertForDataView(mefError))}
            {alertConfigs?.errors && alertConfigs.errors.map((error) => renderTaxReturnAlertForDataView(error))}
            {mefAlertConfigs?.warnings &&
              mefAlertConfigs.warnings.map((mefWarning) => renderMefAlertForDataView(mefWarning))}
            {alertConfigs?.warnings && alertConfigs.warnings.map((warning) => renderTaxReturnAlertForDataView(warning))}
          </div>
        </div>
      </CardBody>

      <CardFooter>
        <div className='df-alt-action'>
          <Link to={detailLink}>
            <Translation i18nKey={`dataviews.review`} collectionId={itemId} />
            <ScreenReaderText />
          </Link>
        </div>
      </CardFooter>
    </Card>
  );
};

export default CollectionItem;
