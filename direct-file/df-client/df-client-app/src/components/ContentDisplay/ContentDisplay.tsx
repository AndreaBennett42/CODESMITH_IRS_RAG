import { CommonContentDisplayProps, CommonContentDisplay } from '@irs/df-common';
import Translation from '../Translation/index.js';
import { useMemo } from 'react';
import { isBeforeResubmissionDeadline } from '../../utils/dateUtils.js';

export type ContentDisplayProps = Omit<CommonContentDisplayProps, 'TranslationComponent'> & {
  allowedTags?: string[] | undefined;
  context?: object;
  className?: string;
};

export const BareContentDisplay = ({ additionalComponents, ...props }: ContentDisplayProps) => {
  // Make sure the customersupportLink is available, if needed
  const innerAdditionalComponents = useMemo(() => {
    const now = new Date();

    return {
      ...additionalComponents,
    };
  }, [additionalComponents]);

  return (
    <CommonContentDisplay
      {...props}
      additionalComponents={innerAdditionalComponents}
      TranslationComponent={Translation}
    />
  );
};

const ContentDisplay = (props: ContentDisplayProps) => {
  return <div className={`screen__info ${props.className}`}>{<BareContentDisplay {...props} />}</div>;
};

export default ContentDisplay;
