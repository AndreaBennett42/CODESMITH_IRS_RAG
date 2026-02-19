import { describe, it, expect } from 'vitest';
import {
  createBooleanWrapper,
  createCollectionWrapper,
  createDollarWrapper,
  createEnumWrapper,
  createTinWrapper,
} from '../persistenceWrappers.js';
import {
  ConcretePath,
  EnumFactory,
  ScalaList,
  convertCollectionToArray,
  scalaListToJsArray,
} from '@irs/js-factgraph-scala';
import { setupFactGraph } from '../setupFactGraph.js';

const uuid1 = `0b1e355e-3d19-415d-8470-fbafd9f58361`;
const uuid2 = `1b1e355e-3d19-415d-8470-fbafd9f58361`;

const twoW2Facts = {
  '/formW2s': createCollectionWrapper([uuid1, uuid2]),
};

const twoFilersFacts = {
  '/filers': createCollectionWrapper([uuid1, uuid2]),
  [`/filers/#${uuid1}/isPrimaryFiler`]: createBooleanWrapper(true),
  [`/filers/#${uuid2}/isPrimaryFiler`]: createBooleanWrapper(false),
  [`/filers/#${uuid1}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
  [`/filers/#${uuid2}/tin`]: createTinWrapper({ area: `222`, group: `22`, serial: `2222` }),
};

const oneFilerFacts = {
  '/filers': createCollectionWrapper([uuid1, uuid2]),
  [`/filers/#${uuid1}/isPrimaryFiler`]: createBooleanWrapper(true),
  [`/filers/#${uuid2}/isPrimaryFiler`]: createBooleanWrapper(false),
  [`/filers/#${uuid1}/tin`]: createTinWrapper({ area: `555`, group: `55`, serial: `5555` }),
};

const MFJFilingStatusFacts = {
  [`/filingStatus`]: createEnumWrapper(`marriedFilingJointly`, `/filingStatusOptions`),
};

const singleFilingStatusFacts = {
  [`/filingStatus`]: createEnumWrapper(`single`, `/filingStatusOptions`),
};

describe(`Combat pay...`, () => {
  describe(`...gets separated between TP and Spouse`, () => {
    it(`/primaryFilerW2s is populated with the primary filer's W2s`, ({ task }) => {
      task.meta.testedFactPaths = [`/primaryFilerW2s`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const primaryFilerW2s = convertCollectionToArray(
        factGraph.get(`/primaryFilerW2s` as ConcretePath).get as ScalaList<string>
      );
      expect(primaryFilerW2s.length).toBe(1);
      expect(primaryFilerW2s[0]).toBe(uuid1);
    });

    it(`/secondaryFilersW2s is populated with the secondary filer's W2s`, ({ task }) => {
      task.meta.testedFactPaths = [`/secondaryFilerW2s`];

      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...MFJFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const secondaryFilerW2s = convertCollectionToArray(
        factGraph.get(`/secondaryFilerW2s` as ConcretePath).get as ScalaList<string>
      );
      expect(secondaryFilerW2s.length).toBe(1);
      expect(secondaryFilerW2s[0]).toBe(uuid2);
    });

    it(`/secondaryFilersW2s does not error when there is no secondary filer`, ({ task }) => {
      task.meta.testedFactPaths = [`/secondaryFilerW2s`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...oneFilerFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const primaryFilerW2s = convertCollectionToArray(
        factGraph.get(`/primaryFilerW2s` as ConcretePath).get as ScalaList<string>
      );

      const secondaryFilerW2s = convertCollectionToArray(
        factGraph.get(`/secondaryFilerW2s` as ConcretePath).get as ScalaList<string>
      );
      expect(primaryFilerW2s.length).toBe(2);
      expect(secondaryFilerW2s.length).toBe(0);
    });

    it(`/MFJSpouseHasCombatPay is true when the MFJ spouse has combat pay`, ({ task }) => {
      task.meta.testedFactPaths = [`/MFJSpouseHasCombatPay`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...MFJFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid2}/writableCombatPay`]: createDollarWrapper(`500`),
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const mfjSpouseHasCombatPay = factGraph.get(`/MFJSpouseHasCombatPay` as ConcretePath).get;
      expect(mfjSpouseHasCombatPay).toBe(true);
    });

    it(`/MFJSpouseHasCombatPay is false when the MFJ spouse has no combat pay`, ({ task }) => {
      task.meta.testedFactPaths = [`/MFJSpouseHasCombatPay`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...MFJFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const mfjSpouseHasCombatPay = factGraph.get(`/MFJSpouseHasCombatPay` as ConcretePath).get;
      expect(mfjSpouseHasCombatPay).toBe(false);
    });

    it(`/MFJSpouseHasCombatPay is false if the filer is not filing jointly`, ({ task }) => {
      task.meta.testedFactPaths = [`/MFJSpouseHasCombatPay`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...singleFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid2}/writableCombatPay`]: createDollarWrapper(`500`),
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const mfjSpouseHasCombatPay = factGraph.get(`/MFJSpouseHasCombatPay` as ConcretePath).get;
      expect(mfjSpouseHasCombatPay).toBe(false);
    });

    it(`/primaryFilerHasCombatPay is true when the primary filer has combat pay`, ({ task }) => {
      task.meta.testedFactPaths = [`/primaryFilerHasCombatPay`];

      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...singleFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid1}/writableCombatPay`]: createDollarWrapper(`500`),
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const primaryFilerW2s = convertCollectionToArray(
        factGraph.get(`/primaryFilerW2s` as ConcretePath).get as ScalaList<string>
      );
      expect(primaryFilerW2s.length).toBe(1);
      const primaryFilerHasCombatPay = factGraph.get(`/primaryFilerHasCombatPay` as ConcretePath).get;
      expect(primaryFilerHasCombatPay).toBe(true);
    });

    it(`/primaryFilerHasCombatPay is false when the primary filer has no combat pay`, ({ task }) => {
      task.meta.testedFactPaths = [`/primaryFilerHasCombatPay`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...singleFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid2}/writableCombatPay`]: createDollarWrapper(`500`),
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const primaryFilerHasCombatPay = factGraph.get(`/primaryFilerHasCombatPay` as ConcretePath).get;
      expect(primaryFilerHasCombatPay).toBe(false);
    });

    it(`Separates the Spouse Combat pay into MFJSpouseCombatPay and primary filer combat pay into /TPCombatPay`, ({
      task,
    }) => {
      task.meta.testedFactPaths = [`/MFJSpouseCombatPay`, `/primaryFilerCombatPay`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...MFJFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid1}/writableCombatPay`]: createDollarWrapper(`300`),
        [`/formW2s/#${uuid2}/writableCombatPay`]: createDollarWrapper(`500`),
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const primaryFilerCombatPay = factGraph.get(`/primaryFilerCombatPay` as ConcretePath).get;
      expect(Number(primaryFilerCombatPay)).toBe(300);
      const MFJSpouseCombatPay = factGraph.get(`/MFJSpouseCombatPay` as ConcretePath).get;
      expect(Number(MFJSpouseCombatPay)).toBe(500);
    });
  });
  describe(`...election options are based on which of the filers have combat pay`, () => {
    it(`When both payers have combat pay, all options are presented`, ({ task }) => {
      task.meta.testedFactPaths = [`/combatPayOptions`];

      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...MFJFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid1}/writableCombatPay`]: createDollarWrapper(`300`),
        [`/formW2s/#${uuid2}/writableCombatPay`]: createDollarWrapper(`500`),
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const combatPayOptions = scalaListToJsArray(
        factGraph.get(`/combatPayOptions` as ConcretePath).get as ScalaList<string>
      );
      expect(combatPayOptions.length).toBe(4);
      expect(combatPayOptions[0]).toBe(`useBothCombatPay`);
      expect(combatPayOptions[1]).toBe(`usePrimaryNotSpouseCombatPay`);
      expect(combatPayOptions[2]).toBe(`useSpouseNotPrimaryCombatPay`);
      expect(combatPayOptions[3]).toBe(`noCombatPay`);
    });
    it(`When only primary filer has combat pay, only primary options are available`, ({ task }) => {
      task.meta.testedFactPaths = [`/combatPayOptions`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...MFJFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid1}/writableCombatPay`]: createDollarWrapper(`300`),
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const combatPayOptions = scalaListToJsArray(
        factGraph.get(`/combatPayOptions` as ConcretePath).get as ScalaList<string>
      );
      expect(combatPayOptions[0]).toBe(`usePrimaryCombatPay`);
      expect(combatPayOptions[1]).toBe(`noCombatPay`);
    });
    it(`When only spouse has combat pay, only spouse options are available`, ({ task }) => {
      task.meta.testedFactPaths = [`/combatPayOptions`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...MFJFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid2}/writableCombatPay`]: createDollarWrapper(`300`),
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const combatPayOptions = scalaListToJsArray(
        factGraph.get(`/combatPayOptions` as ConcretePath).get as ScalaList<string>
      );
      expect(combatPayOptions.length).toBe(2);
      expect(combatPayOptions[0]).toBe(`useSpouseCombatPay`);
      expect(combatPayOptions[1]).toBe(`noCombatPay`);
    });
    it(`When there is no spouse and the primary filer has combat pay, only primary options are available`, ({
      task,
    }) => {
      task.meta.testedFactPaths = [`/combatPayOptions`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...oneFilerFacts,
        ...singleFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid1}/writableCombatPay`]: createDollarWrapper(`300`),
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const combatPayOptions = scalaListToJsArray(
        factGraph.get(`/combatPayOptions` as ConcretePath).get as ScalaList<string>
      );
      expect(combatPayOptions.length).toBe(2);
      expect(combatPayOptions[0]).toBe(`usePrimaryCombatPay`);
      expect(combatPayOptions[1]).toBe(`noCombatPay`);
    });
    it(`When there is no spouse and the primary filer no combat pay, only 'noCombatPay' is available`, ({ task }) => {
      task.meta.testedFactPaths = [`/combatPayOptions`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...oneFilerFacts,
        ...singleFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });

      const primaryFilerW2s = convertCollectionToArray(
        factGraph.get(`/primaryFilerW2s` as ConcretePath).get as ScalaList<string>
      );
      expect(primaryFilerW2s.length).toBe(2);

      const combatPayOptions = scalaListToJsArray(
        factGraph.get(`/combatPayOptions` as ConcretePath).get as ScalaList<string>
      );
      expect(combatPayOptions.length).toBe(1);
    });
    it(`When neither filer has combat pay, only 'noCombatPay' is available`, ({ task }) => {
      task.meta.testedFactPaths = [`/combatPayOptions`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...MFJFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const combatPayOptions = scalaListToJsArray(
        factGraph.get(`/combatPayOptions` as ConcretePath).get as ScalaList<string>
      );
      expect(combatPayOptions.length).toBe(1);
    });
  });

  describe(`Combat pay amount`, () => {
    it(`For MFJ filers, all four enum options work`, ({ task }) => {
      task.meta.testedFactPaths = [`/combatPayAmount`];

      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...MFJFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid1}/writableCombatPay`]: createDollarWrapper(`300`),
        [`/formW2s/#${uuid2}/writableCombatPay`]: createDollarWrapper(`500`),
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      factGraph.set(
        `/combatPayElection` as ConcretePath,
        EnumFactory(`usePrimaryNotSpouseCombatPay`, `/combatPayOptions`).right
      );
      factGraph.save();
      expect(Number(factGraph.get(`/combatPayAmount` as ConcretePath).get)).toBe(300);

      factGraph.set(
        `/combatPayElection` as ConcretePath,
        EnumFactory(`useSpouseNotPrimaryCombatPay`, `/combatPayOptions`).right
      );
      factGraph.save();
      expect(Number(factGraph.get(`/combatPayAmount` as ConcretePath).get)).toBe(500);

      factGraph.set(`/combatPayElection` as ConcretePath, EnumFactory(`useBothCombatPay`, `/combatPayOptions`).right);
      factGraph.save();
      expect(Number(factGraph.get(`/combatPayAmount` as ConcretePath).get)).toBe(800);

      factGraph.set(`/combatPayElection` as ConcretePath, EnumFactory(`noCombatPay`, `/combatPayOptions`).right);
      factGraph.save();
      expect(Number(factGraph.get(`/combatPayAmount` as ConcretePath).get)).toBe(0);
    });

    it(`For single filers, all four enum options work`, ({ task }) => {
      task.meta.testedFactPaths = [`/combatPayAmount`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...oneFilerFacts,
        ...singleFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid1}/writableCombatPay`]: createDollarWrapper(`300`),
        [`/formW2s/#${uuid2}/writableCombatPay`]: createDollarWrapper(`500`),
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      factGraph.set(
        `/combatPayElection` as ConcretePath,
        EnumFactory(`usePrimaryCombatPay`, `/combatPayOptions`).right
      );
      factGraph.save();
      expect(Number(factGraph.get(`/combatPayAmount` as ConcretePath).get)).toBe(800);

      // note: useSpouseCombatPay and useBothCombatPay are not accessible from /combatPayAmount
      factGraph.set(`/combatPayElection` as ConcretePath, EnumFactory(`useSpouseCombatPay`, `/combatPayOptions`).right);
      factGraph.save();
      expect(Number(factGraph.get(`/combatPayAmount` as ConcretePath).get)).toBe(0);

      factGraph.set(`/combatPayElection` as ConcretePath, EnumFactory(`useBothCombatPay`, `/combatPayOptions`).right);
      factGraph.save();
      expect(Number(factGraph.get(`/combatPayAmount` as ConcretePath).get)).toBe(0);

      factGraph.set(`/combatPayElection` as ConcretePath, EnumFactory(`noCombatPay`, `/combatPayOptions`).right);
      factGraph.save();
      expect(Number(factGraph.get(`/combatPayAmount` as ConcretePath).get)).toBe(0);
    });

    it(`When neither filer has combat pay, combat pay is zero`, ({ task }) => {
      task.meta.testedFactPaths = [`/combatPayAmount`];
      const { factGraph } = setupFactGraph({
        ...twoW2Facts,
        ...twoFilersFacts,
        ...MFJFilingStatusFacts,
        [`/formW2s/#${uuid1}/filer`]: {
          item: {
            id: `${uuid1}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
        [`/formW2s/#${uuid2}/filer`]: {
          item: {
            id: `${uuid2}`,
          },
          $type: `gov.irs.factgraph.persisters.CollectionItemWrapper`,
        },
      });
      const combatPayAmount = factGraph.get(`/combatPayAmount` as ConcretePath);
      expect(combatPayAmount.complete).toBe(true);
      expect(combatPayAmount.get.toString()).toBe(`0.00`);
    });
  });
});
