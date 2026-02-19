import { CommonAccordion, CommonAccordionProps } from '@irs/df-common';
import Translation from '../Translation/Translation.js';

interface DFAccordionProps extends CommonAccordionProps {
  splash?: boolean;
}

const DFAccordion = ({ splash = false, ...props }: Omit<DFAccordionProps, 'TranslationComponent'>) => {
  return (
    <div className={splash ? undefined : `margin-top-3`}>
      <CommonAccordion TranslationComponent={Translation} {...props} />
    </div>
  );
};

export default DFAccordion;
