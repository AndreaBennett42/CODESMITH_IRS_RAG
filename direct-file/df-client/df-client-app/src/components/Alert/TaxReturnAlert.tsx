import { FC } from 'react';
import DFAlert, { DFAlertProps } from './DFAlert.js';

export type TaxReturnAlertProps = {
  headingLevel?: DFAlertProps['headingLevel'];
  type: `error` | `warning`;
  factPaths?: string[];
  i18nKey: NonNullable<DFAlertProps[`i18nKey`]>;
  checklistSubcategoryWarningLabel?: string;
} & Omit<DFAlertProps, 'headingLevel' | `i18nKey`>;

const TaxReturnAlert: FC<TaxReturnAlertProps> = ({ headingLevel = `h1`, children, ...props }) => {
  return (
    <DFAlert headingLevel={headingLevel} {...props}>
      {children}
    </DFAlert>
  );
};

export default TaxReturnAlert;
