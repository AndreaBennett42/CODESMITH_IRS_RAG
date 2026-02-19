package gov.irs.directfile.status.services;

import gov.irs.directfile.models.message.Publisher;

public interface StatusChangePublisher extends Publisher {
    // Just a marker interface to allow Spring to inject all the publishers for this message type
}
