import flowNodes from '../../flow/flow.js';
import { createFlowConfig } from '../../flow/flowConfig.js';
import { FlowSetFactInput, checkFlowSetsFacts } from './checkFlowSetsFacts.js';
import { filerWithW2NoDeductionsNoCreditsBaseData } from '../testData.js';

export const uuid = `959c03d1-af4a-447f-96aa-d19397048a44`;
const flow = createFlowConfig(flowNodes);

describe(`Estimated taxes is always completed...`, () => {
  const input: Omit<FlowSetFactInput, 'expectedStartingFactState'> = {
    // eslint-disable-next-line @typescript-eslint/no-non-null-assertion
    startingScreenRoute: flow.subcategoriesByRoute.get(`/flow/your-taxes/estimated-taxes-paid`)!.screens[0].screenRoute,
    startingFactState: filerWithW2NoDeductionsNoCreditsBaseData,
    expectedCompleteFacts: [`/estimatedTaxesIsComplete`],
    collectionId: uuid,
    factValuesToSet: {
      '/paidEstimatedTaxesOrFromLastYear': [true, false],
    },
    leaveOptionalFieldsBlank: false,
  };
  it(`When a filer goes through the section`, () => {
    checkFlowSetsFacts({
      ...input,
      expectedStartingFactState: {},
      startingFactState: input.startingFactState,
    });
  });
});
