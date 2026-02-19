/* eslint-disable max-len */
import { Screen, Subcategory } from '../../flowDeclarations.js';
import {
  DownloadPDFButton,
  Heading,
  InfoDisplay,
  SaveAndOrContinueAndSetFactButton,
  SectionReview,
  Subheading,
} from '../../ContentDeclarations.js';

export const ReviewSubcategory = (
  <Subcategory
    route='review'
    completeIf={[`/hasSeenReviewScreen`, `submissionBlockingFactsAreFalse`]}
    skipDataView={true}
  >
    <Screen route='review' alertAggregatorType='sections'>
      <Heading i18nKey='/heading/complete/review/review-and-confirm' />
      <InfoDisplay i18nKey='/info/complete/review/review-and-confirm' />
      <Subheading i18nKey='subheadings./subheading/complete/review/download-draft' />
      <DownloadPDFButton i18nKey='button.draftPDF' />
      <Subheading i18nKey='subheadings./subheading/complete/review/review-by-section' />
      <SectionReview />
      <SaveAndOrContinueAndSetFactButton
        i18nKey='button.done-reviewing-tax-return'
        sourcePath='/flowTrue'
        destinationPath='/hasSeenReviewScreen'
        shouldBlockOnErrors
      />
    </Screen>
  </Subcategory>
);
