import { useTranslation } from 'react-i18next';
import { IconList as USAIconList, IconListItem, IconListContent, IconListIcon, Icon } from '@trussworks/react-uswds';
import { InfoDisplayProps } from '../../types/core.js';
import Translation from '../Translation/index.js';
import styles from './IconList.module.scss';

const IconList = ({ i18nKey, collectionId }: InfoDisplayProps) => {
  const { t } = useTranslation();
  const translationJson = t(`iconLists.${i18nKey}`, { returnObjects: true });
  if (typeof translationJson !== `object`) {
    return null;
  }
  const listItems = Object.keys(translationJson);
  return (
    <USAIconList className={styles.IconList}>
      {listItems.map((listItemKey, index) => (
        <IconListItem key={index}>
          <IconListIcon>
            <Icon.Check role='presentation' />
          </IconListIcon>
          <IconListContent>
            <Translation i18nKey={`iconLists.${i18nKey}.${listItemKey}`} collectionId={collectionId} />
          </IconListContent>
        </IconListItem>
      ))}
    </USAIconList>
  );
};

export default IconList;
