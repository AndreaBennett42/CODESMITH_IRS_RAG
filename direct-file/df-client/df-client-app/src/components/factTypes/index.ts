import Address from './Address/index.js';
import { BankAccount } from './BankAccount/index.js';
import Boolean from './Boolean/index.js';
import BigContent from '../BigContent/index.js';
import CertifyCheckbox from '../CertifyCheckbox/index.js';
import CollectionDataPreview from '../CollectionDataPreview/CollectionDataPreview.js';
import CollectionItemManager from './CollectionItemManager/index.js';
import CollectionItemReference from './CollectionItemReference/CollectionItemReference.js';
import { ConditionalList } from '../ConditionalList/ConditionalList.js';
import { ContextHeading } from '../ContextHeading/index.js';
import DataPreview from '../DataPreview/DataPreview.js';
import DatePicker from './DatePicker/index.js';
import DFAlert from '../Alert/DFAlert.js';
import DFAccordion from '../Accordion/DFAccordion.js';
import DFModal from '../HelperText/DFModal.js';
import Dollar from './Dollar.js';
import DownloadPDFButton from '../DownloadPDFButton/index.js';
import Ein from './Ein/index.js';
import Enum from './Enum/Enum.js';
import ExitButton from '../ExitButton/index.js';
import FactSelect from '../FormControl/FactSelect/index.js';
import GenericString from './GenericString/index.js';
import Heading from '../Heading.js';
import HelpLink from '../HelperText/HelpLink.js';
import Hint from '../HelperText/Hint.js';
import IconDisplay from '../IconDisplay/IconDisplay.js';
import IconList from '../IconList/index.js';
import InfoDisplay from '../InfoDisplay.js';
import InternalLink from '../InternalLink/index.js';
import IntroContent from '../IntroContent/index.js';
import IpPin from './IpPin/index.js';
import KnockoutButton from '../KnockoutButton/KnockoutButton.js';
import MultiEnum from './MultiEnum/index.js';
import PhoneNumber from './PhoneNumber/index.js';
import Pin from './Pin/index.js';
import SaveAndOrContinueButton from '../SaveAndOrContinueButton/index.js';
import SaveAndOrContinueAndSetFactButton from '../SaveAndOrContinueAndSetFactButton/index.js';
import SectionReview from '../SectionReview/SectionReview.js';
import StateInfoCard from '../StateInfoCard/index.js';
import SubmitButton from '../SubmitButton/index.js';
import Subheading from '../Subheading.js';
import { SummaryTable } from '../SummaryTable/index.js';
import Tin from './Tin/index.js';
import ConditionalAccordion from '../Accordion/ConditionalAccordion.js';
import type {
  CollectionDataViewInternalLinkConfig,
  CollectionItemManagerConfig,
  ConditionalListConfig,
  DFAlertConfig,
  FactConfig,
  IconDisplayConfig,
  InfoDisplayConfig,
  InternalLinkConfig,
  ScreenButtonConfig,
  DataPreviewConfig,
  CollectionDataPreviewConfig,
} from '../../flow/ContentDeclarations.js';
import { FactAssertion, FactResultAssertion } from '../../flow/ContentDeclarations.js';
import CollectionDataViewInternalLink from '../InternalLink/CollectionDataViewInternalLink.js';
import LimitingString from './LimitingString/LimitingString.js';
import StateTaxesButton from '../StateTaxesButton/StateTaxesButton.js';
import FileYourStateTaxesDetails from '../FileYourStateTaxesDetails/FileYourStateTaxesDetails.js';
import StateTaxReminderAlertWrapper from '../StateInfoAlert/StateTaxReminderAlertWrapper.js';

export const FactTypeRenderer = {
  Address,
  Boolean,
  BankAccount,
  CollectionItemReference,
  DatePicker,
  Dollar,
  Ein,
  Enum,
  MultiEnum,
  FactSelect,
  GenericString,
  IpPin,
  LimitingString,
  PhoneNumber,
  Pin,
  Tin,
} satisfies Record<FactConfig[`componentName`], unknown>;

export const CollectionItemManagerRenderer = { CollectionItemManager } satisfies Record<
  CollectionItemManagerConfig[`componentName`],
  unknown
>;

export const IconDisplayRenderer = { IconDisplay } satisfies Record<IconDisplayConfig[`componentName`], unknown>;

export const DFAlertRenderer = { DFAlert } satisfies Record<DFAlertConfig[`componentName`], unknown>;

export const ConditionalListRenderer = { ConditionalList, SummaryTable } satisfies Record<
  ConditionalListConfig[`componentName`],
  unknown
>;

export const InternalLinkRenderer = { InternalLink } satisfies Record<InternalLinkConfig[`componentName`], unknown>;
export const DataPreviewRenderer = { DataPreview } satisfies Record<DataPreviewConfig[`componentName`], unknown>;
export const CollectionDataPreviewRenderer = { CollectionDataPreview } satisfies Record<
  CollectionDataPreviewConfig[`componentName`],
  unknown
>;

export const CollectionDataViewInternalLinkRenderer = { CollectionDataViewInternalLink } satisfies Record<
  CollectionDataViewInternalLinkConfig[`componentName`],
  unknown
>;

export const InfoTypeRenderer = {
  BigContent,
  CertifyCheckbox,
  ContextHeading,
  DFModal,
  DFAccordion,
  ConditionalAccordion,
  Heading,
  HelpLink,
  Hint,
  IconList,
  InfoDisplay,
  IntroContent,
  SectionReview,
  StateInfoCard,
  StateTaxReminderAlertWrapper,
  Subheading,
  FactAssertion,
  FactResultAssertion,
  FileYourStateTaxesDetails,
} satisfies Record<InfoDisplayConfig[`componentName`], unknown>;

export const ScreenButtonRenderer = {
  DownloadPDFButton,
  ExitButton,
  KnockoutButton,
  SaveAndOrContinueButton,
  SaveAndOrContinueAndSetFactButton,
  SubmitButton,
  StateTaxesButton,
} satisfies Record<ScreenButtonConfig[`componentName`], unknown>;
