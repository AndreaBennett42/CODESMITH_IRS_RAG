# Static site for Direct File landing page & screener

The Direct File landing page and screener is a set of pages that users see as the entry point before logging in to file their taxes.

It is coded in React but is 100% static. There are no API calls.

## Development

Get started by running `npm install` and `npm start` from the [df-static-site/](./) directory (this one!)

### Pausing the pilot

Set the [CURRENT_PHASE](./src/constants.ts) variable (in [src/constants.ts](./src/constants.ts)), following the directions in that file.

:warning: This will not close the Direct File application to new tax filers! :warning:

Instead, it changes the copy on the landing page and removes the public-facing chat button.
