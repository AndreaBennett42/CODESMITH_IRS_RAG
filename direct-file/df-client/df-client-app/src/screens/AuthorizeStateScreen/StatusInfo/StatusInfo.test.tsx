import StatusInfo, { StatusInfoProps } from './StatusInfo.js';
import { render, screen } from '@testing-library/react';
import { FEDERAL_RETURN_STATUS } from '../../../constants/taxConstants.js';
import { within } from '@testing-library/dom';

describe(`StatusInfo`, () => {
  const renderStatusInfo = (props: StatusInfoProps) => {
    render(<StatusInfo {...props} />);

    const statusInfo = screen.getByRole(`alert`);

    return {
      statusInfo,
    };
  };

  it(`Renders for status ${FEDERAL_RETURN_STATUS.ACCEPTED}`, () => {
    const { statusInfo } = renderStatusInfo({ taxReturnStatus: FEDERAL_RETURN_STATUS.ACCEPTED });

    expect(statusInfo).toHaveTextContent(FEDERAL_RETURN_STATUS.ACCEPTED);
  });

  it(`Renders for status ${FEDERAL_RETURN_STATUS.REJECTED}`, () => {
    const { statusInfo } = renderStatusInfo({ taxReturnStatus: FEDERAL_RETURN_STATUS.REJECTED });

    expect(statusInfo).toHaveTextContent(FEDERAL_RETURN_STATUS.REJECTED);
  });

  it(`Renders for status ${FEDERAL_RETURN_STATUS.PENDING}`, () => {
    const { statusInfo } = renderStatusInfo({ taxReturnStatus: FEDERAL_RETURN_STATUS.PENDING });

    expect(statusInfo).toHaveTextContent(FEDERAL_RETURN_STATUS.PENDING);
  });

  it(`Renders with child content`, () => {
    const content = `Information about the status!`;

    const { statusInfo } = renderStatusInfo({
      taxReturnStatus: FEDERAL_RETURN_STATUS.PENDING,
      children: <>{content}</>,
    });

    const childContent = within(statusInfo).queryByText(content);

    expect(statusInfo).toHaveTextContent(FEDERAL_RETURN_STATUS.PENDING);
    expect(childContent).toBeInTheDocument();
  });
});
