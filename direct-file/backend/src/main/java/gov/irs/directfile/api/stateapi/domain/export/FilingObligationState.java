package gov.irs.directfile.api.stateapi.domain.export;

import java.util.Arrays;
import java.util.List;

public enum FilingObligationState {
    ARIZONA("Arizona", "az"),
    CALIFORNIA("California", "ca"),
    CONNECTICUT("Connecticut", "ct"),
    IDAHO("Idaho", "ia"),
    ILLINOIS("Illinois", "il"),
    KANSAS("Kansas", "ka"),
    MAINE("Maine", "me"),
    MARYLAND("Maryland", "md"),
    MASSACHUSETTS("Massachusetts", "ma"),
    NORTH_CAROLINA("North Carolina", "nc"),
    NEW_JERSEY("New Jersey", "nj"),
    NEW_MEXICO("New Mexico", "nm"),
    NEW_YORK("New York", "ny"),
    OREGON("Oregon", "or"),
    PENNSYLVANIA("Pennsylvania", "pa"),
    WISCONSIN("Wisconsin", "wi");

    private final String name;
    private final String abbreviation;

    FilingObligationState(String name, String abbreviation) {
        this.name = name;
        this.abbreviation = abbreviation;
    }

    public static List<String> abbreviations() {
        return Arrays.stream(values()).map(state -> state.abbreviation).toList();
    }

    public static List<String> names() {
        return Arrays.stream(values()).map(state -> state.name).toList();
    }
}
