import { ConcretePath } from '@irs/js-factgraph-scala';
import {
  CollectionControlDeclaration,
  FactDeclaration,
  InfoDisplayDeclaration,
  SaveAndOrContinueAndSetFactButtonDeclaration,
  ScreenButtonDeclaration,
} from '../flow/ContentDeclarations.js';
import { Dispatch, OptionHTMLAttributes, ReactNode, SetStateAction } from 'react';
import { FEDERAL_RETURN_STATUS } from '../constants/taxConstants.js';

/**
 * Below are the props that Screen.tsx will use to render the screens.
 * So if Screen.tsx needs to populate a prop, it needs to be declared here.
 * (If not, it may be declared in the component file itself.)
 * All screen component props should extend an associated FC<ComponentNameDeclaration>.
 */
export type FactProps = Omit<FactDeclaration, 'required'> & {
  concretePath: ConcretePath;
  isValid: boolean;
  onValidData: (path: ConcretePath, validity: boolean) => void;
  showFeedback: boolean;
  collectionId: string | null;
  required?: boolean;
  saveAndPersist: () => Promise<{ hasPersistError: boolean }>;
  factRefs?: React.MutableRefObject<Map<string, React.MutableRefObject<HTMLInputElement>>>;
  handleFactRef?: (concretePath: ConcretePath) => React.MutableRefObject<HTMLInputElement>;
  factValidity?: Map<ConcretePath, boolean>;
};

export type BooleanFactProps = FactProps & {
  inputType?: 'radio' | 'checkbox';
  i18nKeySuffixContext?: string;
};

export type EnumFactProps = FactProps & {
  labelledBy?: 'heading' | 'legend';
  renderAs?: 'radio' | 'select';
  i18nKeySuffixContext?: string;
  skipBlank?: true;
};

export type MultiEnumProps = FactProps & {
  labelledBy?: 'heading' | 'legend';
  i18nKeySuffixContext?: string;
};

export type DatePickerFactProps = FactProps & {
  lockYearTo?: ConcretePath;
  lastAllowableDatePath?: ConcretePath;
  disallowPastDates?: boolean;
};

export type AddressFactProps = FactProps & {
  useCombinedStreetLengthForValidation?: true;
};

export type InfoDisplayProps = InfoDisplayDeclaration & {
  collectionId: string | null;
  showFeedback?: boolean;
  gotoNextScreen?: () => void;
  handleChange?: (event: React.ChangeEvent<HTMLInputElement>) => void;
  inlineButtonI18nKey?: string;
};

export type ScreenButtonProps = ScreenButtonDeclaration & {
  collectionId: string | null;
  i18nKey?: string;
  showFeedback?: boolean;
  hasCertified?: boolean;
  gotoNextScreen: () => void;
  screenHasFacts: boolean;
  factValidity: Map<ConcretePath, boolean>;
  setShowFeedback: Dispatch<SetStateAction<boolean>>;
  focusOnErrorOrSummary: () => void;
  taxId?: string;
};

export type SaveAndOrContinueAndSetFactButtonProps = SaveAndOrContinueAndSetFactButtonDeclaration & {
  gotoNextScreen: () => void;
  collectionId: string | null;
  shouldBlockOnErrors?: boolean;
};

export type CollectionLoopControlProps = FactProps & CollectionControlDeclaration;

export type FetchArgs = [input: RequestInfo | URL, init?: RequestInit | undefined];

export type Fetch = (input: RequestInfo | URL, init?: RequestInit | undefined) => Promise<Response>;

export interface FetchResponse<T> {
  success: boolean;
  data: T;
  loading: boolean;
  callRead: () => void;
}

export interface UseSaveResponse<T> extends Omit<FetchResponse<T>, 'callRead'> {
  save: () => void;
}

export type UseFetchPdfResponse = Omit<FetchResponse<null>, `data` | 'callRead'> & {
  fetchPdf: (taxId: string) => void;
};

export interface AuthAPIResponse {
  accessToken: string;
  userId: string;
  email: string;
  url: string;
}

export interface LoginApiResponse {
  url: string;
}

export type FederalReturnStatus = (typeof FEDERAL_RETURN_STATUS)[keyof typeof FEDERAL_RETURN_STATUS];

// Note: TaxReturn values that change over time are only updated within the client when re-fetching returns
export interface TaxReturn {
  id: string;
  createdAt: string;
  taxYear: number;
  facts: { [path: string]: FactValue };
  taxReturnSubmissions: TaxReturnSubmission[];
  isEditable: boolean;
  dataImportBehavior?: string;
  surveyOptIn: boolean | null;
}

export interface TaxReturnSubmission {
  id: string;
  submitUserId: string;
  createdAt: string;
  receiptId: string | null;
  submissionReceivedAt: string | null;
}

// If receiptId and submissionReceivedAt are null, the submission has been recently submitted but not registered
// within the status app yet, so no up-to-date status is available. Otherwise, the submission is "ready" for status
// to be retrieved
export interface ReceivedTaxReturnSubmission extends TaxReturnSubmission {
  receiptId: string;
  submissionReceivedAt: string;
}

export interface TaxReturnSubmissionStatus {
  status: FederalReturnStatus;
  rejectionCodes: RejectedStatus[];
  createdAt: string;
}

export interface RejectedStatus {
  MeFErrorCode: string;
  TranslationKey: string;
  MeFDescription: string;
}

// TODO: this should be a union of values exported from
// js-factgraph-scala, but it seems like there's some need
// for a manual serde layer right now.
export interface FactValue {
  $type: string;
  item: unknown;
}

export interface AuthPII {
  email: string;
}

export interface SelectOption<T extends OptionHTMLAttributes<never>[`value`]> {
  label: ReactNode;
  value: T;
}
