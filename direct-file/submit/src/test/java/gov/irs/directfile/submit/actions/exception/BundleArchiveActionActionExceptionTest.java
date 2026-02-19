package gov.irs.directfile.submit.actions.exception;

import java.util.List;

import org.junit.jupiter.api.Test;

import gov.irs.directfile.audit.events.TinType;
import gov.irs.directfile.submit.domain.UserContextData;

import static org.junit.jupiter.api.Assertions.*;

class BundleArchiveActionActionExceptionTest {

    @Test
    public void
            whenUserContextDataTaxReturnIdsToString_thenShouldReturnStringContainingTaxReturnIdsFromUserContextDataList() {
        UserContextData userContextData1 = new UserContextData("", "", "111", "", TinType.INDIVIDUAL, "", "");
        UserContextData userContextData2 = new UserContextData("", "", "222", "", TinType.INDIVIDUAL, "", "");
        List<UserContextData> userContextDataList = List.of(userContextData1, userContextData2);
        BundleArchiveActionException bundleArchiveActionException =
                new BundleArchiveActionException(userContextDataList, null, new Exception());

        String result = bundleArchiveActionException.userContextDataTaxReturnIdsToString();

        assertEquals("UserContextData{TaxReturnIds=[111, 222]}", result);
    }
}
