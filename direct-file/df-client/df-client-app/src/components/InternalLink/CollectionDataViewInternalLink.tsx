import Translation from '../Translation/index.js';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import { Link } from 'react-router-dom';
import { CommonTranslation } from 'df-i18n';
import { FC } from 'react';
import { useTranslation } from 'react-i18next';
import { Path } from '../../flow/Path.js';
import { CollectionDataViewInternalLinkDeclaration } from '../../flow/ContentDeclarations.js';

export type CollectionDataViewInternalLinkProps = CollectionDataViewInternalLinkDeclaration;

const CollectionDataViewInternalLink: FC<CollectionDataViewInternalLinkProps> = ({
  dataViewUrl,
  i18nKey,
  collectionItemPath,
}) => {
  const { factGraph } = useFactGraph();
  const { i18n } = useTranslation();
  const nameSpacedKey = CommonTranslation.getNamespacedKey(i18nKey);
  const fullI18nKey = `${nameSpacedKey}.internalLink`;
  if (!i18n.exists(fullI18nKey)) {
    return null;
  }
  const collectionItemId = factGraph.get(Path.concretePath(collectionItemPath, null)).get.idString;
  const url = `${dataViewUrl}/${collectionItemId}/`;
  return (
    <Translation
      i18nKey={fullI18nKey}
      collectionId={collectionItemId}
      components={{ InternalLink: <Link to={url}></Link> }}
    />
  );
};

export default CollectionDataViewInternalLink;
