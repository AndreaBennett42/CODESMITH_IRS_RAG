package gov.irs.directfile.status.domain;

import java.util.List;
import java.util.stream.Collectors;

import lombok.Getter;

import gov.irs.mef.AcknowledgementList;

@Getter
public class AcknowledgementsListWrapper {
    private final List<AcknowledgementWrapper> acknowledgements;

    public AcknowledgementsListWrapper(AcknowledgementList acknowledgementList) {
        this.acknowledgements = List.copyOf(acknowledgementList.getAcknowledgements().stream()
                .map(AcknowledgementWrapper::fromMefAcknowledgement)
                .collect(Collectors.toList()));
    }

    public AcknowledgementsListWrapper(List<AcknowledgementWrapper> acknowledgementWrappers) {
        this.acknowledgements = List.copyOf(acknowledgementWrappers);
    }
}
