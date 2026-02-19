import { useContext, FC } from 'react';
import { NetworkConnectionContext } from '../../context/networkConnectionContext.js';
import Translation from '../Translation/index.js';
import { ScreenButtonProps } from '../../types/core.js';
import { CommonLinkRenderer } from 'df-common-link-renderer';
import classnames from 'classnames';

const ExitButton: FC<ScreenButtonProps> = ({ collectionId }) => {
  const { online } = useContext(NetworkConnectionContext);

  const linkClasses = classnames(`usa-button`, { 'usa-button--disabled': !online });

  return (
    <div className='screen__actions'>
      <CommonLinkRenderer className={linkClasses} url='/home'>
        <Translation i18nKey={`button.dashboard`} collectionId={collectionId} />
      </CommonLinkRenderer>
    </div>
  );
};

export default ExitButton;
