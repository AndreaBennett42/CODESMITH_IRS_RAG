import { ESLintUtils } from '@typescript-eslint/utils';

/* We don't publish our rules, but eslint wants a url, so we've left this as the default url for now.*/
const createRule = ESLintUtils.RuleCreator((name) => `https://example.com/rule/${name}`);

export const noFactGraphSaveRule = createRule({
  create(context) {
    return {
      CallExpression(node) {
        const parserServices = context.parserServices;
        if (!parserServices) {
          return;
        }
        const checker = parserServices.program.getTypeChecker();
        const esTreeNodeToTSNodeMap = parserServices.esTreeNodeToTSNodeMap;
        if (node.callee.type === `MemberExpression`) {
          const tsNode = esTreeNodeToTSNodeMap.get(node.callee.object);
          const type = checker.getTypeAtLocation(tsNode);
          const typeName = checker.typeToString(type);

          if (
            node.callee.property.type === `Identifier` &&
            node.callee.property.name === `save` &&
            typeName === `FactGraph`
          ) {
            context.report({
              node,
              messageId: `default`,
            });
          }
        }
      },
    };
  },
  meta: {
    docs: {
      recommended: `error`,
      description: `Don't save to the factgraph without persisting it.`,
    },
    messages: {
      // eslint-disable-next-line max-len
      default: `You probably shouldn't be doing a .save without persisting the fact graph. Do you want to use saveAndPersist instead?`,
    },
    type: `suggestion`,
    schema: [],
  },
  name: `No fact graph save`,
  defaultOptions: [],
});
