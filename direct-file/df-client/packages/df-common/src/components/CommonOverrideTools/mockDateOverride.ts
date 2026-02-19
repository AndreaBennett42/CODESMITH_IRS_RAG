export function overrideDateIfRequired() {
  if (process.env.NODE_ENV === `development`) {
    // capture the original Date.now function before it's overridden
    const originalDateNow = Date.now;

    const overrideTimeString = sessionStorage.getItem(`df_date_override`);
    const simulatedTime = overrideTimeString ? new Date(overrideTimeString).getTime() : null;
    const timeDifference = simulatedTime !== null ? simulatedTime - originalDateNow() : 0;

    class MockedDate extends Date {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      constructor(...args: any[]) {
        /* the compiler just would not accept (...args) used with a constructor, so because
        this is just dev convenience, we manually handle extra arguments and call it a day*/
        if (args.length === 0) {
          super();
          if (simulatedTime !== null) {
            this.setTime(originalDateNow() + timeDifference);
          }
        } else if (args.length === 1) {
          super(args[0]);
        } else if (args.length === 2) {
          super(args[0], args[1]);
        } else if (args.length === 3) {
          super(args[0], args[1], args[2]);
        } else if (args.length === 4) {
          super(args[0], args[1], args[2], args[3]);
        } else if (args.length === 5) {
          super(args[0], args[1], args[2], args[3], args[4]);
        } else if (args.length === 6) {
          super(args[0], args[1], args[2], args[3], args[4], args[5]);
        } else if (args.length === 7) {
          super(args[0], args[1], args[2], args[3], args[4], args[5], args[6]);
        } else {
          // More than seven arguments, fallback to just using the current time
          super();
        }
      }

      static now() {
        return originalDateNow() + timeDifference;
      }
    }

    MockedDate.UTC = Date.UTC;
    MockedDate.parse = Date.parse;

    window.Date = MockedDate as unknown as DateConstructor;
  }
}
