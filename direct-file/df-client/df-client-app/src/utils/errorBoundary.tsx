import { Component, ReactNode, ErrorInfo } from 'react';
import DFAlert from '../components/Alert/DFAlert.js';
import { connect, ConnectedProps } from 'react-redux';

interface State {
  hasError: boolean;
}

// We could use the error info interface but this would give us type errors if react changes that
// which I think is a benefit.
export type ReactErrorPayload = {
  componentStack: string | null;
  digest: string | null;
} | null;

class ErrorBoundary extends Component<ErrorBoundaryProps, State> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  componentDidCatch = async (inputError: Error, errorInfo: ErrorInfo) => {
    this.setState({ hasError: true });
    const { componentStack, digest } = errorInfo;
    const payload: ReactErrorPayload = {
      componentStack: componentStack || null,
      digest: digest || null,
    };
  };

  render() {
    if (this.state.hasError) {
      return (
        <DFAlert
          type='error'
          i18nKey='errorBoundary.siteWide'
          collectionId={null}
          headingLevel='h3'
          className={`margin-y-3`}
        />
      );
    }
    return this.props.children;
  }
}

const connector = connect(null);
type PropsFromRedux = ConnectedProps<typeof connector>;

// Component props type
type ErrorBoundaryProps = PropsFromRedux & {
  children: ReactNode;
};

// Export the connected component
export default connector(ErrorBoundary);
