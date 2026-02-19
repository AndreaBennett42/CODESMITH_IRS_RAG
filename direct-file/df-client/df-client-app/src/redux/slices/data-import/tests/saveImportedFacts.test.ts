import { saveImportedFacts } from '../saveImportedFacts.js';
import { setupFactGraph } from '../../../../test/setupFactGraph.js';
import { createFlowConfig } from '../../../../flow/flowConfig.js';
import { contentConfigIsFactConfig } from '../../../../flow/ContentDeclarations.js';
import flowNodes from '../../../../flow/flow.js';
import { testDataImportProfileConfig } from '../DevDataImportConfig.js';
import { Path } from '../../../../flow/Path.js';
import { CollectionFactory, ConcretePath } from '@irs/js-factgraph-scala';

const primaryFilerId = `b5090897-f4e7-4eac-9230-957f40637efe`;
const baseFilerData = {
  '/email': { $type: `gov.irs.factgraph.persisters.EmailAddressWrapper`, item: { email: `user.0000@example.com` } },
  '/filers': {
    $type: `gov.irs.factgraph.persisters.CollectionWrapper`,
    item: { items: [`b5090897-f4e7-4eac-9230-957f40637efe`] },
  },
  '/filers/#b5090897-f4e7-4eac-9230-957f40637efe/tin': {
    $type: `gov.irs.factgraph.persisters.TinWrapper`,
    item: { area: `123`, group: `45`, serial: `6788` },
  },
  '/filers/#b5090897-f4e7-4eac-9230-957f40637efe/isPrimaryFiler': {
    $type: `gov.irs.factgraph.persisters.BooleanWrapper`,
    item: true,
  },
};

describe(`saveImportedFacts`, () => {
  Object.entries(testDataImportProfileConfig).forEach(([simpsonsCharacter, profileGetter]) => {
    describe(`For ${simpsonsCharacter} `, () => {
      const profile = profileGetter();
      const { factGraph } = setupFactGraph(baseFilerData);
      const flow = createFlowConfig(flowNodes);
      const profilesWithDisallowedName = [`kent`, `abe`, `december27`];
      const profilesWithBadAddresses = [`kent`];
      const profileHasDisallowedName = profilesWithDisallowedName.includes(simpsonsCharacter);
      const profileHasBadAddress = profilesWithBadAddresses.includes(simpsonsCharacter);

      it(`can save imported facts in about you successfully`, () => {
        const aboutYou = flow.subcategoriesByRoute.get(`/flow/you-and-your-family/about-you`);

        if (aboutYou) {
          const facts = aboutYou.screens.flatMap((s) =>
            s.content.filter(contentConfigIsFactConfig).filter((c) => c.props.path && c.props.importedPath)
          );

          saveImportedFacts(profile, factGraph, facts, primaryFilerId);

          const firstName = factGraph.get(Path.concretePath(`/primaryFiler/firstName`, primaryFilerId));
          const middleInitial = factGraph.get(Path.concretePath(`/primaryFiler/middleInitial`, primaryFilerId));
          const lastName = factGraph.get(Path.concretePath(`/primaryFiler/lastName`, primaryFilerId));
          const dob = factGraph.get(Path.concretePath(`/primaryFiler/dateOfBirth`, primaryFilerId));
          const phone = factGraph.get(Path.concretePath(`/phone`, primaryFilerId));
          const address = factGraph.get(Path.concretePath(`/address`, primaryFilerId));

          if (!profileHasDisallowedName) {
            expect(firstName.hasValue).toBe(true);
            expect(firstName.get).not.toBe(``);
            expect(lastName.get).not.toBe(``);
          } else {
            expect(firstName.hasValue).toBe(false);
            expect(lastName.hasValue).toBe(false);
          }

          if (profile.data.aboutYouBasic.state === `success`) {
            expect(middleInitial.hasValue).toBe(true);
            if (profile.data.aboutYouBasic.payload.middleInitial) {
              expect(middleInitial.get).not.toBe(``);
            }

            if (profile.data.aboutYouBasic.payload.dateOfBirth) {
              expect(dob.hasValue).toBe(true);
            }
            if (profile.data.aboutYouBasic.payload.mobileNumber) {
              expect(phone.hasValue).toBe(true);
            }
            if (!profileHasBadAddress) {
              expect(address.hasValue).toBe(true);
            }
          }
        }
      });

      it(`can save imported facts in jobs successfully`, () => {
        const w2Context = `/formW2s`;
        const w2sLoop = flow.collectionLoopsByName.get(w2Context);

        if (w2sLoop) {
          if (profile.data.w2s.state === `success`) {
            const newIds = profile.data.w2s.payload.map((w2) => w2.id);
            factGraph.set(w2Context as ConcretePath, CollectionFactory(newIds));
            factGraph.save();
            const facts = w2sLoop.screens.flatMap((s) =>
              s.content.filter(contentConfigIsFactConfig).filter((c) => c.props.path && c.props.importedPath)
            );

            if (profile.data.w2s.payload.length) {
              profile.data.w2s.payload.forEach((w2) => {
                saveImportedFacts(profile, factGraph, facts, w2.id);
                const ein = factGraph.get(Path.concretePath(`/formW2s/*/ein`, w2.id));
                const wages = factGraph.get(Path.concretePath(`/formW2s/*/writableWages`, w2.id));

                expect(ein.hasValue).toBe(true);
                expect(wages.hasValue).toBe(true);
                expect(Number(wages.get.toString())).toBe(Number(w2.wagesTipsOtherCompensation));
              });
            }
          }
        }
      });
    });
  });
});
