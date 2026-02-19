# Exported `directFileData` Snapshots

This directory contains snapshot artifacts from tests that verify the output of fact graph data that gets exported to states alongside the MeF XML

> [!WARNING]
> If you notice unintentional diffs on many of/all the snapshot files, there's a good chance you've just broken the state-api!
> 
> Please read the [FAQ](#faq) below to understand the impact of your changes before proceeding.
> 
> If you are still unsure, ask the state integration team.

## FAQ

### What do changes to these snapshots translate to in terms of application behavior?

Each snapshot represents the payload value that would be exported to states for a given scenario. Any diffs introduced
by code changes would mean state partners would see equivalent diffs after deployment of said changes.

### DirectFileDataScenarioSnapshotTest failed after I made some changes. Now what?

Simply running the tests locally and committing the changes to the snapshot files will allow them to pass.
Please read the answer to the next question to determine if the snapshot changes are expected or indicative of a problem.

### What do I do if my changes trigger updates to these snapshots?

Generally, adding a new scenario will most commonly trigger updates to this directory, but should be isolated to a single file.
In that case, there's nothing to worry about. Thank you for adding to the wealth of scenarios that help keep this feature highly tested!

If you've added a newly exported field, you should see that field show up in accordance with the documentation for every scenario.
This is good, because your tests have updated themselves and prompted review.

### What is a "breaking" change?

Thinking of the snapshots as example HTTP response bodies, any change that alters the "API contract" in a way that would
break consumer parsing of the response is a breaking change.

For example, the following are breaking changes:
* Renaming a fact that is exported will change the name of the key in the exported response 
* Deleting a fact that is exported, or removing the `stateSystems` export will remove data from the response payload

Breaking changes are explicitly not supported and will result in incident with all state consumers of the payload,
and corresponding remediation.
