import { ScreenButtonDeclaration, SummaryTableDeclaration } from './ContentDeclarations.js';
import { findFlowNodesOfType } from './findFlowNodesOfType.js';
import { findFlowNodesWithProp } from './findFlowNodesWithProp.js';
import flowNodes from './flow.js';
import { createFlowConfig } from './flowConfig.js';

describe(`Flow Nodes corresponding internal links`, () => {
  it(`All SummaryTable InternalLink nodes have corresponding screens`, () => {
    const summaryTables = findFlowNodesOfType(flowNodes, `SummaryTable`);
    const itemsWithLinks = summaryTables.flatMap((st) =>
      (st.props as SummaryTableDeclaration).items.filter((i) => i.internalLink)
    );
    const allScreens = createFlowConfig(flowNodes).screensByRoute;
    for (const listItem of itemsWithLinks) {
      if (listItem.internalLink && !allScreens.has(listItem.internalLink)) {
        throw new Error(`ListItem contained invalid internal link ${listItem.internalLink}`);
      }
    }
  });

  it(`All SaveAndOrContinueButton nextRouteOverride nodes have corresponding screens`, () => {
    const saveAndContinueButtons = findFlowNodesOfType(flowNodes, `SaveAndOrContinueButton`);
    const buttonsWithRouteOverrides = saveAndContinueButtons
      .filter((st) => (st.props as ScreenButtonDeclaration).nextRouteOverride !== undefined)
      .map((st) => st.props as ScreenButtonDeclaration);
    const allScreens = createFlowConfig(flowNodes).screensByRoute;
    const topLevelPageRoutes = [`/checklist`, `/data-view`];
    for (const buttonRoute of buttonsWithRouteOverrides) {
      const routeDoesNotHaveScreenNode = topLevelPageRoutes.some((v) => buttonRoute.nextRouteOverride?.includes(v));
      if (
        buttonRoute.nextRouteOverride &&
        !allScreens.has(buttonRoute.nextRouteOverride) &&
        !routeDoesNotHaveScreenNode
      ) {
        throw new Error(`SaveAndOrContinueButton contained invalid route override ${buttonRoute.nextRouteOverride}`);
      }
    }
  });

  it(`All nodes with edit routes have corresponding edit screens`, () => {
    const nodesWithEditRoutes = findFlowNodesWithProp(flowNodes, `editRoute`);
    const allScreens = createFlowConfig(flowNodes).screensByRoute;
    for (const n of nodesWithEditRoutes) {
      if (n.props.editRoute && !allScreens.has(n.props.editRoute)) {
        throw new Error(`Node contained invalid edit link ${n.props.editRoute}`);
      }
    }
  });

  it(`All nodes with knockoutRoute routes have corresponding edit screens`, () => {
    const nodesWithKnockoutRoutes = findFlowNodesWithProp(flowNodes, `knockoutRoute`);
    const allScreens = createFlowConfig(flowNodes).screensByRoute;
    for (const n of nodesWithKnockoutRoutes) {
      if (n.props.knockoutRoute && !allScreens.has(n.props.knockoutRoute)) {
        throw new Error(`Node contained invalid knockoutRoute path ${n.props.knockoutRoute}`);
      }
    }
  });
});
