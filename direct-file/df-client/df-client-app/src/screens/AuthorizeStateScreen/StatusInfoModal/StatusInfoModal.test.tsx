import StatusInfoModal, { StatusInfoModalProps } from './StatusInfoModal.js';
import { render, screen } from '@testing-library/react';
import { createRef } from 'react';
import { ModalRef } from '@trussworks/react-uswds';
import { within } from '@testing-library/dom';

describe(StatusInfoModal.name, () => {
  const renderStatusInfoModal = (props: StatusInfoModalProps) => {
    render(<StatusInfoModal {...props} />);

    const modal = screen.getByRole(`dialog`);
    const heading = within(modal).getByRole(`heading`);
    const modalCloseButton = within(modal).getByRole(`button`, { name: `Close this window` });

    return { modal, heading, modalCloseButton };
  };

  it(`Renders without error`, () => {
    const modalRef = createRef<ModalRef>();

    const { modal, heading, modalCloseButton } = renderStatusInfoModal({ modalRef, canTransfer: true });

    expect(modal).toHaveClass(`is-hidden`);
    expect(heading).toHaveTextContent(`heading`);
    expect(modalCloseButton).toBeInTheDocument();
  });

  it(`Renders without error when canTransfer is false`, () => {
    const modalRef = createRef<ModalRef>();

    const { modal, heading, modalCloseButton } = renderStatusInfoModal({ modalRef, canTransfer: false });

    expect(modal).toHaveClass(`is-hidden`);
    expect(heading).toHaveTextContent(`heading`);
    expect(modalCloseButton).toBeInTheDocument();
  });
});
