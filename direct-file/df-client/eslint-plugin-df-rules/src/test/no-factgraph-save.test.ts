import { RuleTester } from '@typescript-eslint/utils/dist/ts-eslint';
import { noFactGraphSaveRule } from '../no-factgraph-save.js';
import { describe, it } from 'vitest';

const ruleTester = new RuleTester({
  parser: require.resolve(`@typescript-eslint/parser`),
});

/**
 * The test runner runs natively with mocha, but to use vitest, we have
 * to wrap it.
 */
describe(`Test runner`, () => {
  it(`no-factgraph-save`, () => {
    ruleTester.run(`no-factgraph-save`, noFactGraphSaveRule, {
      valid: [
        {
          code: `
          const factGraph = {} as FactGraph;
          factGraph.get();
        `,
        },
      ],
      invalid: [
        {
          code: `
          const factGraph = {} as FactGraph;
          factGraph.save();
        `,
          errors: [
            {
              messageId: `default`,
            },
          ],
        },
      ],
    });
  });
});
