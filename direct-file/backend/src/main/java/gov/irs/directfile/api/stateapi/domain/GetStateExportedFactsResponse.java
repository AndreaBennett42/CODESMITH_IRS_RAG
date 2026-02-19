package gov.irs.directfile.api.stateapi.domain;

import gov.irs.directfile.api.stateapi.domain.export.ExportableFacts;

public record GetStateExportedFactsResponse(ExportableFacts exportedFacts) {}
