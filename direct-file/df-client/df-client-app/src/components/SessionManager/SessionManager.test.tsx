import { render, screen, waitFor } from '@testing-library/react';

import { SessionManager } from './SessionManager.js';
import { SessionClock } from '../../auth/session.js';

describe(`SessionManager`, () => {
  // TODO: flaky test (https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/2590)
  test.skip(`alertUser and closeAlert open and close the modal`, async () => {
    render(
      <div>
        <SessionManager />
      </div>
    );
    const clock = new SessionClock();
    const alert = screen.getByRole(`dialog`, { name: `auth.timeout.header` });
    expect(alert).toHaveClass(`is-hidden`);
    window.sessionManager.alertUser(clock);
    // test failure is here
    waitFor(() => expect(alert).toHaveClass(`is-visible`));
    window.sessionManager.closeAlert();
    waitFor(() => expect(alert).toHaveClass(`is-hidden`));
  });
});
