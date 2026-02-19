import { RefObject, Component, createRef } from 'react';
import { ButtonGroup, Modal, ModalFooter, ModalHeading, ModalRef, ModalToggleButton } from '@trussworks/react-uswds';
import { Trans, withTranslation, WithTranslation } from 'react-i18next';
import { SessionClock } from '../../auth/session.js';

export interface _SessionManager {
  alertUser: (clock: SessionClock) => void;
  warnUser: (clock: SessionClock) => void;
  closeAlert: () => void;
}

/**
 * Pop open a modal to warn tax filers of past or impending session termination.
 */
export class _SessionManager extends Component<WithTranslation, { clock: SessionClock }> {
  // user is going to be logged out
  warningModalRef: RefObject<ModalRef>;
  // user has already been logged out
  goodbyeModalRef: RefObject<ModalRef>;

  constructor(props: WithTranslation) {
    super(props);
    this.warningModalRef = createRef<ModalRef>();
    this.goodbyeModalRef = createRef<ModalRef>();
  }

  /**
   * Open the warning modal.
   *
   * This method can be called anywhere in the JavaScript codebase,
   * using `window.SessionManager.warnUser()`.
   */
  warnUser = (clock: SessionClock) => {
    this.setState({ clock: clock });
    this.warningModalRef.current?.toggleModal?.(undefined, true);
  };

  /**
   * Open the alert modal.
   *
   * This method can be called anywhere in the JavaScript codebase,
   * using `window.SessionManager.alertUser()`.
   */
  alertUser = (clock: SessionClock) => {
    this.setState({ clock: clock });
    this.goodbyeModalRef.current?.toggleModal?.(undefined, true);
  };

  /**
   * Close either modal.
   *
   * This method can be called anywhere in the JavaScript codebase,
   * using `window.SessionManager.closeAlert()`.
   */
  closeAlert = () => {
    this.warningModalRef.current?.toggleModal?.(undefined, false);
    this.goodbyeModalRef.current?.toggleModal?.(undefined, false);
  };

  handleExtend = () => {
    this.state.clock?.extend?.();
  };

  handleSignOut = () => {
    this.state.clock?.terminate?.();
  };

  render() {
    const t = this.props.t;
    return (
      <>
        {/* Modal for alerting user that they've been signed out */}
        <Modal
          ref={this.goodbyeModalRef}
          forceAction
          aria-labelledby='session-timed-out-heading'
          aria-describedby='session-timed-out-description'
          id='session-timed-out'
        >
          <ModalHeading id='session-timed-out-heading'>{t(`auth.timed-out.header`)}</ModalHeading>
          <div className='usa-prose'>
            <p id='session-timed-out-description'>
              <Trans>{t(`auth.timed-out.body`)}</Trans>
            </p>
          </div>
          <ModalFooter>
            <ButtonGroup>
              <ModalToggleButton modalRef={this.goodbyeModalRef} closer onClick={this.handleSignOut}>
                {t(`auth.timed-out.signIn`)}
              </ModalToggleButton>
            </ButtonGroup>
          </ModalFooter>
        </Modal>
        {/* Modal for asking user if they'd like to extend their session */}
        <Modal
          ref={this.warningModalRef}
          forceAction
          aria-labelledby='session-timeout-heading'
          aria-describedby='session-timeout-description'
          id='session-timeout'
        >
          <ModalHeading id='session-timeout-heading'>{t(`auth.timeout.header`)}</ModalHeading>
          <div className='usa-prose'>
            <p id='session-timeout-description'>
              <Trans>{t(`auth.timeout.body`)}</Trans>
            </p>
          </div>
          <ModalFooter>
            <ButtonGroup>
              <ModalToggleButton modalRef={this.warningModalRef} closer onClick={this.handleExtend}>
                {t(`auth.timeout.extend`)}
              </ModalToggleButton>
              <ModalToggleButton
                modalRef={this.warningModalRef}
                closer
                onClick={this.handleSignOut}
                unstyled
                className='padding-105 text-center'
              >
                {t(`auth.timeout.signOut`)}
              </ModalToggleButton>
            </ButtonGroup>
          </ModalFooter>
        </Modal>
      </>
    );
  }
}

declare global {
  interface Window {
    sessionManager: _SessionManager;
  }
}

/**
 * Ensure that SessionManager component instances are accessible via `window.sessionManager`.
 *
 * This higher order component does nothing special, it is merely a convenience wrapper,
 * as otherwise the ref would need to be supplied by the parent using `<SessionManager />`.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function withRef(Component: any) {
  return function WrappedComponent() {
    return (
      <Component
        ref={(sessionManager: _SessionManager) => {
          window.sessionManager = sessionManager;
        }}
      />
    );
  };
}

export const SessionManager = withRef(withTranslation(undefined, { withRef: true })(_SessionManager));
