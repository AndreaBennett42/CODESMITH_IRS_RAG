import { CommonContentDisplay } from '@irs/df-common';
import { Icon, IconListContent, IconListIcon, IconListItem } from '@trussworks/react-uswds';

import { Translation } from '../../components/index.js';

interface DFIconListItemProps {
  i18nKey: string;
  mood: 'good' | 'bad';
  hasHTML?: boolean;
}

const DFIconListItem = ({ i18nKey, mood, hasHTML }: DFIconListItemProps) => {
  const useGreen = mood === `good`;
  const icon = useGreen ? <Icon.CheckCircle /> : <Icon.Cancel />;

  return (
    <IconListItem>
      <IconListIcon className={useGreen ? `text-green` : `text-red`} aria-hidden='true'>
        {icon}
      </IconListIcon>
      <IconListContent>
        <CommonContentDisplay i18nKey={i18nKey} noBody={!hasHTML} TranslationComponent={Translation} />
      </IconListContent>
    </IconListItem>
  );
};

export default DFIconListItem;
