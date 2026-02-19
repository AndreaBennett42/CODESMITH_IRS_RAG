import { QuietErrorMessage } from '../QuietErrorMessage.js';
import Hint from '../HelperText/Hint.js';
import DFModal from '../HelperText/DFModal.js';
import { buildHintId, buildHintKey, buildReadonlyHintKey } from './helpers.js';
import { I18nKey } from '../Translation/Translation.js';
import { Path } from '../../fact-dictionary/Path.js';

export interface FormFieldWrapperWithConfigurableLabelProps {
  path?: Path;
  hintKey?: I18nKey;
  name?: string;
  collectionId: string | null;
  controlId: string;
  hintId?: string;
  showError: boolean;
  errorMessage?: string | JSX.Element;
  useLegendAsLabel?: boolean;
  children: JSX.Element | JSX.Element[];
  readonly?: boolean;
  required?: boolean;
  label: JSX.Element;
}

export function FormFieldWrapperWithConfigurableLabel({
  path,
  hintKey,
  name,
  collectionId,
  controlId,
  hintId,
  showError,
  errorMessage,
  children,
  readonly,
  label,
}: FormFieldWrapperWithConfigurableLabelProps) {
  if (!hintKey) {
    const defaultHintKey = buildHintKey(path, name);
    const readonlyHintKey = buildReadonlyHintKey(path);

    hintKey = readonly ? readonlyHintKey : defaultHintKey;
  }

  if (!hintId) {
    hintId = buildHintId(controlId);
  }

  return (
    <>
      {label}
      {path && <DFModal i18nKey={hintKey} collectionId={collectionId} />}
      {path && !readonly && <Hint hintId={hintId} i18nKey={hintKey} collectionId={collectionId} />}
      {showError && <QuietErrorMessage id={`${controlId}__error-msg`}>{errorMessage}</QuietErrorMessage>}
      {children}
    </>
  );
}
