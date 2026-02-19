import {
  buildCustomFilingDeadlineDate,
  formatAsDateTimeString,
  getDaysBetween,
  getFullDaysBetween,
  isBetweenFilingDeadlineAndPerfectionDeadline,
  MILLISECONDS_PER_DAY,
} from './dateUtils.js';
import {
  END_OF_FILING_DEADLINE,
  END_OF_PERFECTION_DEADLINE,
  FILING_DEADLINE,
  PERFECTION_DEADLINE,
} from '../constants/taxConstants.js';

describe(getDaysBetween.name, () => {
  it(`calculates the number of days between two dates correctly`, () => {
    const first = new Date(Date.parse(`1862-07-01`));
    const second = new Date(Date.parse(`1862-07-05`));
    const daysBetween = getDaysBetween(first, second);

    expect(daysBetween).toEqual(4);
  });
  it(`calculates the number of days between two dates in different months correctly`, () => {
    const first = new Date(Date.parse(`1862-07-01`));
    const second = new Date(Date.parse(`1862-08-01`));
    const daysBetween = getDaysBetween(first, second);

    expect(daysBetween).toEqual(31);
  });
  it(`returns 0 when the days are the same day`, () => {
    const first = new Date(Date.parse(`1862-07-01`));
    const daysBetween = getDaysBetween(first, first);

    expect(daysBetween).toEqual(0);
  });
  it(`returns a negative number when the second date is past the first (also works across month boundaries)`, () => {
    const first = new Date(Date.parse(`1862-07-01`));
    const second = new Date(Date.parse(`1862-06-30`));
    const daysBetween = getDaysBetween(first, second);

    expect(daysBetween).toEqual(-1);
  });
  it(`Returns correct number of days between configured period before the configured filing deadline`, () => {
    const daysBeforeFilingDeadline = 4;
    const first = new Date(FILING_DEADLINE.getTime() - daysBeforeFilingDeadline * MILLISECONDS_PER_DAY);

    const daysBetween = getDaysBetween(first, FILING_DEADLINE);

    expect(daysBetween).toEqual(4);
  });
});

describe(getFullDaysBetween.name, () => {
  it(`Returns correct number of full days between two dates, if partial days are involved`, () => {
    const daysBeforeFilingDeadline = 4.9;
    const first = new Date(FILING_DEADLINE.getTime() - daysBeforeFilingDeadline * MILLISECONDS_PER_DAY);

    const daysBetween = getFullDaysBetween(first, FILING_DEADLINE);

    expect(daysBetween).toEqual(4);
  });
});

describe(buildCustomFilingDeadlineDate.name, () => {
  it(`outputs the correct value`, () => {
    // April 17th, we don't care about the year, test with 23:59:59 EDT
    const customFilingDeadline = `0000-04-17 23:59:59.999999`;

    const customFilingDeadlineDate = buildCustomFilingDeadlineDate(customFilingDeadline);

    expect(customFilingDeadlineDate.getFullYear()).toEqual(new Date().getFullYear());
    expect(customFilingDeadlineDate.getMonth()).toEqual(3);
    expect(customFilingDeadlineDate.getDate()).toEqual(17);
    expect(customFilingDeadlineDate.getHours()).toEqual(23);
    expect(customFilingDeadlineDate.getMinutes()).toEqual(59);
    expect(customFilingDeadlineDate.getSeconds()).toEqual(59);
    expect(customFilingDeadlineDate.getMilliseconds()).toEqual(999);
  });
});

describe(formatAsDateTimeString.name, () => {
  it(`formats to the correct length`, () =>
    expect(formatAsDateTimeString(`en-US`, new Date(Date.parse(`2024-1-1 1:20:30 AM`))).length).toEqual(
      `MM/DD/YYYY HH:MM:SS`.length
    ));
});

describe(isBetweenFilingDeadlineAndPerfectionDeadline.name, () => {
  it.each([END_OF_FILING_DEADLINE, PERFECTION_DEADLINE])(`returns true for dates within the range: %s`, (date) => {
    const result = isBetweenFilingDeadlineAndPerfectionDeadline(date);

    expect(result).toBeTruthy();
  });

  it.each([FILING_DEADLINE, END_OF_PERFECTION_DEADLINE])(`returns false for dates not within the range: %s`, (date) => {
    const result = isBetweenFilingDeadlineAndPerfectionDeadline(date);

    expect(result).toBeFalsy();
  });
});
