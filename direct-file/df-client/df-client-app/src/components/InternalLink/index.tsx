import Translation from '../Translation/index.js';
import { Path as FGPath } from '../../fact-dictionary/Path.js';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { getCollectionItems } from '../../hooks/useCollectionItems.js';
import { Link } from 'react-router-dom';
import { useFlow } from '../../flow/flowConfig.js';
import { CommonTranslation } from 'df-i18n';
import { FC } from 'react';
import { useTranslation } from 'react-i18next';
import { hasOwn } from '../../utils/polyfills.js';

export type InternalLinkProps = {
  i18nKey: string;
  collectionId: string | null;
  route: string;
  displayAsButton?: boolean;
};

const InternalLink: FC<InternalLinkProps> = ({ route, i18nKey, collectionId, displayAsButton }) => {
  const { factGraph } = useFactGraph();
  const { screensByRoute } = useFlow();
  const { i18n } = useTranslation();
  const nameSpacedKey = CommonTranslation.getNamespacedKey(i18nKey);
  const fullI18nKey = `${nameSpacedKey}.internalLink`;
  if (!i18n.exists(fullI18nKey)) {
    return null;
  }
  const linkedScreen = screensByRoute.get(route);
  const collectionName = linkedScreen?.collectionContext as FGPath;
  const collectionNameHasAlias = collectionName
    ? hasOwn(factGraph.getDictionary().getDefinition(collectionName).value, `getAlias`)
    : false;
  const collection =
    collectionName && collectionNameHasAlias ? getCollectionItems(collectionName, collectionId, factGraph) : null;
  const routeCollectionId = collectionId || collection?.collectionItems[0];
  // Check if the route needs a collection ID in the URL
  const url = collectionName && routeCollectionId && linkedScreen ? linkedScreen.fullRoute(routeCollectionId) : route;
  const buttonClass = displayAsButton ? `usa-button` : ``;
  return (
    <Translation
      i18nKey={fullI18nKey}
      collectionId={collectionId}
      components={{ InternalLink: <Link className={`df-internal-link ${buttonClass}`} to={url}></Link> }}
    />
  );
};

export default InternalLink;
