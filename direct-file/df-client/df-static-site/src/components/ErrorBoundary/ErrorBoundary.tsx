import { Component, ComponentType, ReactNode } from 'react';
import { withTranslation, WithTranslation } from 'react-i18next';
import { Alert } from '@trussworks/react-uswds';

interface Props extends WithTranslation {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  componentDidCatch(error: Error) {
    this.setState({ hasError: true });
  }

  render() {
    if (this.state.hasError) {
      const { t } = this.props;
      return (
        <Alert
          className='margin-top-3'
          type='error'
          heading={t(`components.errorBoundary.siteWide.alertText.heading`)}
          headingLevel='h3'
          validation
        >
          <p>{t(`components.errorBoundary.siteWide.alertText.body.text`)}</p>
        </Alert>
      );
    }
    return this.props.children;
  }
}
const hoisted: ComponentType<{ children: ReactNode }> = withTranslation(`translation`)(ErrorBoundary);
export default hoisted;
