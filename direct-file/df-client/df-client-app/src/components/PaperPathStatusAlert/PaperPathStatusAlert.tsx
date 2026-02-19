import { useTranslation } from 'react-i18next';
import { Alert } from '@trussworks/react-uswds';

const PaperPathStatusAlert = () => {
  const { t } = useTranslation(`translation`);

  return (
    <Alert
      type='warning'
      headingLevel='h3'
      heading={t(`paperPathStatusAlert.heading`)}
      data-testid='paper-path-status-alert'
    >
      {t(`paperPathStatusAlert.body`)}
    </Alert>
  );
};

export default PaperPathStatusAlert;
