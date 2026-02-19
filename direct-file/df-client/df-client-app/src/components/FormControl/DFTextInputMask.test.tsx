import { maskedWithCursorPosition, strippedWithCursorIndex } from './DFTextInputMask.js';

describe(`strippedWithCursorIndex`, () => {
  it(`should strip non-dash characters and return the correct cursor position`, () => {
    const result = strippedWithCursorIndex(`abc-123`, 4);
    expect(result).toEqual({ stripped: `abc123`, cursorPosition: 3 });
  });

  it(`should handle leading dashes correctly`, () => {
    const result = strippedWithCursorIndex(`--123`, 2);
    expect(result).toEqual({ stripped: `123`, cursorPosition: 0 });
  });

  it(`should handle double dashes correctly`, () => {
    const result = strippedWithCursorIndex(`abc--123`, 6);
    expect(result).toEqual({ stripped: `abc123`, cursorPosition: 4 });
  });

  it(`should handle null input correctly`, () => {
    const result = strippedWithCursorIndex(`abc--123`, null);
    expect(result).toEqual({ stripped: `abc123`, cursorPosition: 6 });
  });
});

describe(`maskedWithCursorPosition`, () => {
  it(`should apply the mask correctly and return the correct cursor position`, () => {
    const result = maskedWithCursorPosition(`abc123`, `__-__-_`, 3);
    expect(result).toEqual({ masked: `ab-c1-2`, cursorPosition: 4 });
  });

  it(`should handle leading mask characters`, () => {
    const result = maskedWithCursorPosition(`abc123`, `--___-___`, 4);
    expect(result).toEqual({ masked: `--abc-123`, cursorPosition: 7 });
  });

  it(`should truncate values that are too long`, () => {
    const result = maskedWithCursorPosition(`abc1234`, `___-___`, 0);
    expect(result).toEqual({ masked: `abc-123`, cursorPosition: 0 });
  });

  it(`should bump the cursor after the mask character`, () => {
    const result = maskedWithCursorPosition(`abc123`, `___-___`, 3);
    expect(result).toEqual({ masked: `abc-123`, cursorPosition: 4 });
  });

  it(`should not show remaining pigeon hole characters if the value isn't long enough`, () => {
    const result = maskedWithCursorPosition(`abc12`, `___-___`, 3);
    expect(result).toEqual({ masked: `abc-12`, cursorPosition: 4 });
  });

  it(`should not show trailing mask characters if the value isn't long enough`, () => {
    const result = maskedWithCursorPosition(`abc`, `___-___`, 1);
    expect(result).toEqual({ masked: `abc`, cursorPosition: 1 });
  });
});
