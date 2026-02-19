import { InfoDisplayProps } from '../../types/core.js';
import { FC } from 'react';
import { Button, Icon } from '@trussworks/react-uswds';
import { Condition, RawCondition } from '../../flow/Condition.js';
import { useFactGraph } from '../../factgraph/FactGraphContext.js';
import Translation from '../Translation/index.js';
import { IconName } from '../IconDisplay/IconDisplay.js';

const ContinueButton: FC<InfoDisplayProps & { enabled?: RawCondition; iconName?: IconName }> = ({
  gotoNextScreen,
  enabled,
  i18nKey,
  collectionId,
  iconName,
}) => {
  const { factGraph } = useFactGraph();
  const isDisabled = enabled && !new Condition(enabled).evaluate(factGraph, collectionId);
  const IconComponent = iconName ? Icon[iconName] : undefined;
  return (
    <div className='screen__actions'>
      <Button disabled={isDisabled} type='submit' onClick={gotoNextScreen}>
        {IconComponent && <IconComponent />}
        <span>
          <Translation i18nKey={i18nKey} collectionId={collectionId} />
        </span>
      </Button>
    </div>
  );
};

export default ContinueButton;
