# Scenarios

This directory contains a large collection of test scenarios. These scenarios
are JSON files that contain client tax return submissions. The JSON is in the
format of a [`SubmitRequestBody`][1] object.

## Adding Scenarios

If you are starting with a fact graph JSON from the client that you obtained
using `debugFactGraph.download()`, you can run it through
`direct-file/scripts/make-scenario.py`

If you create the scenario JSON via other means, please sort the keys to make
manual inspection and future diffs easier. You can use this command from this `scenarios` directory (you may need to `brew install sponge`):

```sh
jq --indent 2 --sort-keys . -- "my-scenario.json" | sponge -- "my-scenario.json"
```

to reformat the file in place.

Before commiting your scenario, be sure to run prettier to reformat the JSON
since this directory is symlinked into `df-client/df-client-app/` subdirectories
`src/test/factDictionaryTests` and `src/test/scenarioTests`.

```sh
# from the project root
cd direct-file/df-client
npm run format-write
```

