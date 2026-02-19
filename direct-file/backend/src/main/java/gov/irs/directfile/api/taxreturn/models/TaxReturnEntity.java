package gov.irs.directfile.api.taxreturn.models;

import java.util.Map;
import java.util.UUID;

import gov.irs.directfile.models.FactTypeWithItem;

public interface TaxReturnEntity {
    UUID getId();

    String getFactsCipherText();

    void setFactsCipherText(String cipherText);

    Map<String, FactTypeWithItem> getFacts();

    void setFactsWithoutDirtyingEntity(Map<String, FactTypeWithItem> facts);

    // Field "store" is part of TaxReturn, not TaxReturnSubmission
    default void setStoreCipherText(String store) {}

    default void setStoreWithoutDirtyingEntity(String store) {}

    default String getStoreCipherText() {
        return null;
    }

    default String getStore() {
        return null;
    }
}
