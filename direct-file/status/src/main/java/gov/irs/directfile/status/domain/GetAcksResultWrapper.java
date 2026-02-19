package gov.irs.directfile.status.domain;

import lombok.Getter;

import gov.irs.mef.services.transmitter.mtom.GetAcksResult;

@Getter
public class GetAcksResultWrapper {
    private AcknowledgementsListWrapper acknowledgementsListWrapper;

    public GetAcksResultWrapper(GetAcksResult getAcksResult) {
        this.acknowledgementsListWrapper = new AcknowledgementsListWrapper(getAcksResult.getAcknowledgementList());
    }

    public GetAcksResultWrapper(AcknowledgementsListWrapper acknowledgementsListWrapper) {
        this.acknowledgementsListWrapper = acknowledgementsListWrapper;
    }
}
