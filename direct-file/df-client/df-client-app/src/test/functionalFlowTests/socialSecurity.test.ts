import { it, describe, expect } from 'vitest';
import { baseFilerData } from '../testData.js';
import { createFlowConfig } from '../../flow/flowConfig.js';
import flowNodes from '../../flow/flow.js';
import makeGivenFacts from './functionalFlowUtils.js';
import { setupFactGraph } from '../setupFactGraph.js';

const flow = createFlowConfig(flowNodes);
const givenFacts = makeGivenFacts(flow);

describe(`The Social Security income loop`, () => {
  const path = `/flow/income/social-security`;
  // for (const scenario of [
  //   { form: `SSA-1099`, screen: `ssa-whose` },
  //   { form: `RRB-1099`, screen: `rrb-whose` },
  // ]) {
  //   describe(`When the income is from form ${scenario.form} and the filer is married filing jointly`, () => {
  //     const reportId = `0b1e355e-3d19-415d-8470-fbafd9f58361`;
  //     const { factGraph } = setupFactGraph({
  //       ...baseFilerData,
  //       [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
  //       '/socialSecurityReports': {
  //         $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
  //         item: { items: [`${reportId}`] },
  //       },
  //       [`/socialSecurityReports/#${reportId}/formType`]: createEnumWrapper(
  //         scenario.form,
  //         `/socialSecurityIncomeFormTypeOptions`
  //       ),
  //     });
  //     it(`navigates to screen ${scenario.screen}`, ({ task }) => {
  //       expect(givenFacts(factGraph).atPath(`${path}/ssa-or-rrb-choice`, reportId, task)).toRouteNextTo(
  //         `${path}/${scenario.screen}`
  //       );
  //     });
  //   });
  // }

  for (const scenario of [
    { form: `SSA-1099`, screen: `ssa-1099-box-5` },
    // { form: `RRB-1099`, screen: `rrb-1099-box-5` },
  ]) {
    describe(`When the income is from form ${scenario.form}`, () => {
      const reportId = `0b1e355e-3d19-415d-8470-fbafd9f58361`;
      const { factGraph } = setupFactGraph({
        ...baseFilerData,
        '/socialSecurityReports': {
          $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
          item: { items: [`${reportId}`] },
        },
        // [`/socialSecurityReports/#${reportId}/formType`]: createEnumWrapper(
        //   scenario.form,
        //   `/socialSecurityIncomeFormTypeOptions`
        // ),
      });
      it(`navigates to screen ${scenario.screen}`, ({ task }) => {
        expect(givenFacts(factGraph).atPath(`${path}/ssa-whose`, reportId, task)).toRouteNextTo(
          `${path}/${scenario.screen}`
        );
      });
    });
  }

  it(`moves from ssa-1099-box-5 to ssa-1099-box-6`, ({ task }) => {
    const reportId = `0b1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/socialSecurityReports': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${reportId}`] },
      },
      // [`/socialSecurityReports/#${reportId}/formType`]: createEnumWrapper(
      //   `SSA-1099`,
      //   `/socialSecurityIncomeFormTypeOptions`
      // ),
    });

    expect(givenFacts(factGraph).atPath(`${path}/ssa-1099-box-5`, reportId, task)).toRouteNextTo(
      `${path}/ssa-1099-box-6`
    );
  });

  // it(`moves from rrb-1099-box-5 to rrb-1099-box-10`, ({ task }) => {
  //   const reportId = `0b1e355e-3d19-415d-8470-fbafd9f58361`;
  //   const { factGraph } = setupFactGraph({
  //     ...baseFilerData,
  //     '/socialSecurityReports': {
  //       $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
  //       item: { items: [`${reportId}`] },
  //     },
  //     [`/socialSecurityReports/#${reportId}/formType`]: createEnumWrapper(
  //       `RRB-1099`,
  //       `/socialSecurityIncomeFormTypeOptions`
  //     ),
  //   });

  //   expect(givenFacts(factGraph).atPath(`${path}/rrb-1099-box-5`, reportId, task)).toRouteNextTo(
  //     `${path}/rrb-1099-box-10`
  //   );
  // });

  it(`moves from ssa-1099-box-6 to data view of social security`, ({ task }) => {
    const reportId = `0b1e355e-3d19-415d-8470-fbafd9f58361`;
    const { factGraph } = setupFactGraph({
      ...baseFilerData,
      '/socialSecurityReports': {
        $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
        item: { items: [`${reportId}`] },
      },
      // [`/socialSecurityReports/#${reportId}/formType`]: createEnumWrapper(
      //   `SSA-1099`,
      //   `/socialSecurityIncomeFormTypeOptions`
      // ),
    });

    expect(givenFacts(factGraph).atPath(`${path}/ssa-1099-box-6`, reportId, task)).toRouteNextTo(
      `/data-view/loop/%2FsocialSecurityReports/${reportId}`
    );
  });

  // it(`moves from rrb-1099-box-10 to data view of social security`, ({ task }) => {
  //   const reportId = `0b1e355e-3d19-415d-8470-fbafd9f58361`;
  //   const { factGraph } = setupFactGraph({
  //     ...baseFilerData,
  //     '/socialSecurityReports': {
  //       $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
  //       item: { items: [`${reportId}`] },
  //     },
  //     [`/socialSecurityReports/#${reportId}/formType`]: createEnumWrapper(
  //       `RRB-1099`,
  //       `/socialSecurityIncomeFormTypeOptions`
  //     ),
  //   });

  //   expect(givenFacts(factGraph).atPath(`${path}/rrb-1099-box-10`, reportId, task)).toRouteNextTo(
  //     `/data-view/loop/%2FsocialSecurityReports/${reportId}`
  //   );
  // });
});
