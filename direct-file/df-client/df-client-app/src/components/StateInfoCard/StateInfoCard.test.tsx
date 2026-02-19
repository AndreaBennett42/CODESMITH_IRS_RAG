import { REF_LOCATION, REF_LOCATION_VALUE } from '../../constants/pageConstants.js';
import { appendQueryParams } from './StateInfoCard.js';

describe(`appendQueryParams`, () => {
  it(`append ref_location to Url`, () => {
    const url = new URL(`https://sth.url`);
    appendQueryParams(url);

    expect(url.searchParams.size).toBe(1);
    expect(url.searchParams.get(REF_LOCATION)).toBe(REF_LOCATION_VALUE.SUBMISSION);
  });
});
