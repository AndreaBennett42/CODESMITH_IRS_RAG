import { render, waitFor } from '@testing-library/react';
import { within } from '@testing-library/dom';
import PageTitle from './PageTitle.js';
import { wrapComponent } from '../../test/helpers.js';

describe(`PageTitle`, () => {
  test(`renders H1`, () => {
    const title = `Title`;
    const { getByRole } = render(wrapComponent(<PageTitle redactedTitle='redacted-title'>{title}</PageTitle>));
    const heading = getByRole(`heading`, { level: 1 });
    expect(heading).toBeInTheDocument();
    expect(within(heading).getByText(title)).toBeInTheDocument();
  });

  test(`redacted title shown as page title`, async () => {
    document.title = ``; //reset title
    const title1 = `Test title 1`;
    const redactedTitle = `redacted-title`;
    render(
      wrapComponent(
        <main id='main' tabIndex={-1}>
          <PageTitle redactedTitle='redacted-title'>{title1}</PageTitle>
        </main>
      )
    );
    const main = document.getElementById(`main`);
    expect(main).toHaveFocus();

    //wait for update to document.title
    await waitFor(() => {
      expect(document.title).toBe(`${redactedTitle}`);
    });
  });

  test(`focuses on <main> on title change`, async () => {
    document.title = ``; //reset title
    const title1 = `Test title 1`;
    const redactedTitle = `redacted-title`;
    const { rerender } = render(
      wrapComponent(
        <main id='main' tabIndex={-1}>
          <PageTitle redactedTitle={redactedTitle}>{title1}</PageTitle>
        </main>
      )
    );
    const main = document.getElementById(`main`);

    expect(main).toHaveFocus();

    //wait for update to document.title
    await waitFor(() => {
      expect(document.title).toBe(`${redactedTitle}`);
    });

    //remove focus on main
    main?.blur();

    //rerender page with same PageTitle
    rerender(
      wrapComponent(
        <main id='main' tabIndex={-1}>
          <PageTitle redactedTitle='redacted-title'>{title1}</PageTitle>
        </main>
      )
    );
    expect(main).not.toHaveFocus();

    const title2 = `Test title 2`;
    //TODO trigger location change
    rerender(
      wrapComponent(
        <main id='main' tabIndex={-1}>
          <PageTitle redactedTitle='redacted-title'>{title2}</PageTitle>
        </main>
      )
    );

    // Skip until useLocation() is mocked;
    // await waitFor(() => {
    //   expect(document.title).toBe(`${title2}`);
    //   expect(main).toHaveFocus();
    // });
  });

  it.skip(`document.title is updated on location change`, async () => {
    document.title = ``; //reset title
    const title1 = `Test title 1`;
    const { rerender } = render(
      wrapComponent(
        <>
          <PageTitle redactedTitle='redacted-title'>{title1}</PageTitle>
        </>
      )
    );
    await waitFor(() => expect(document.title).toBe(`${title1}`));

    const title2 = `Test title 2`;
    rerender(
      wrapComponent(
        <>
          <PageTitle redactedTitle='redacted-title'>{title2}</PageTitle>
        </>
      )
    );
    await waitFor(() => expect(document.title).toBe(`${title2}`));
  });
});
