import { testDataImportProfileConfig } from '../DevDataImportConfig.js';

const profilesWithW2ParseErrors = [`krusty`, `sean`, `bart`, `lisa`, `kent`, `w2_parse_error`];

describe(`The Data Import Test Configs`, () => {
  Object.entries(testDataImportProfileConfig).forEach(([simpsonsCharacter, profileGetter]) => {
    describe(`Config: ${simpsonsCharacter}`, () => {
      const profile = profileGetter();
      Object.entries(profile.data).forEach(([payloadType, payload]) => {
        const profileHasW2ParseError = profilesWithW2ParseErrors.includes(simpsonsCharacter) && payloadType === `w2s`;
        it(`should not have a parse error for ${payloadType} payload`, () => {
          if (payload.state === `error` && !profileHasW2ParseError) {
            expect(payload.errorType).not.toEqual(`parse-error`);
          }
        });
      });
      it(`should have ${simpsonsCharacter} with all w2s having 00 in the middle of the ein`, () => {
        const { w2s } = profile.data;

        if (w2s.state !== `success`) {
          return;
        }
        const eins = w2s.payload.map((w2) => w2.ein);
        eins.forEach((ein) => {
          expect(ein.startsWith(`00`)).toBeTruthy();
        });
      });
    });
  });
});
