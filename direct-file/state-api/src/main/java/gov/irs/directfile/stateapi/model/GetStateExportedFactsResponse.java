package gov.irs.directfile.stateapi.model;

import java.util.Map;

// TODO: Consolidate exported facts related classes/records between backend and state-api
//       in shared data-models dependency https://git.irslabs.org/irslabs-prototypes/direct-file/-/issues/9570
public record GetStateExportedFactsResponse(Map<String, Object> exportedFacts) {}
