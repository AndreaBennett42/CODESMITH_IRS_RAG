import { Alert, Button, GridContainer } from '@trussworks/react-uswds';
import StackedButtonGroup from '../../../components/StackedButtonGroup/StackedButtonGroup.js';
import { MouseEventHandler, ReactNode } from 'react';
import { useTranslation } from 'react-i18next';
import SystemAlertAggregator from '../../../components/SystemAlertAggregator/SystemAlertAggregator.js';

export type ErrorScreenProps = {
  children?: ReactNode;
  errorMessage: ReactNode;
  handleGoBack: MouseEventHandler<HTMLButtonElement>;
};

const ErrorScreen = ({ children, errorMessage, handleGoBack }: ErrorScreenProps) => {
  const { t } = useTranslation(`translation`, { keyPrefix: `authorizeState.errorScreen` });

  return (
    <GridContainer>
      <SystemAlertAggregator />
      <Alert type='error' headingLevel='h2' role='alert'>
        {errorMessage}
      </Alert>
      {children}
      <StackedButtonGroup>
        <Button type='button' unstyled onClick={handleGoBack}>
          {t(`goBack`)}
        </Button>
      </StackedButtonGroup>
    </GridContainer>
  );
};

export default ErrorScreen;
