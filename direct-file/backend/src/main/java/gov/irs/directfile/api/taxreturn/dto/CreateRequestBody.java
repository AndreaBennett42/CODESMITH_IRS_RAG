package gov.irs.directfile.api.taxreturn.dto;

import java.util.Map;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;

import gov.irs.directfile.models.FactTypeWithItem;

@Getter
@Setter
public class CreateRequestBody {
    @NotNull @Min(value = 2023)
    @Max(value = 2050)
    private int taxYear;

    @NotNull(message = "No facts provided") private Map<@NotEmpty String, @NotNull FactTypeWithItem> facts;

    public static final String docsExampleObject =
            """
    {
        "taxYear": 2024,
        "facts": {
            "/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/canBeClaimed": {
                "$type": "gov.irs.factgraph.persisters.BooleanWrapper",
                "item": false
            },
            "/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/occupation": {
                "$type": "gov.irs.factgraph.persisters.StringWrapper",
                "item": "mat"
            },
            "/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/middleInitial": {
                "$type": "gov.irs.factgraph.persisters.StringWrapper",
                "item": "R"
            },
            "/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/employerName": {
                "$type": "gov.irs.factgraph.persisters.StringWrapper",
                "item": "the USG"
            },
            "/formW2s": {
                "$type": "gov.irs.factgraph.persisters.CollectionWrapper",
                "item": {
                    "items": [
                        "3d1946aa-7280-43d4-b5c9-5fde6a6ba28c"
                    ]
                }
            },
            "/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/medicareWithholding": {
                "$type": "gov.irs.factgraph.persisters.DollarWrapper",
                "item": "5000.00"
            },
            "/taxYear": {
                "$type": "gov.irs.factgraph.persisters.IntWrapper",
                "item": 2022
            },
            "/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/oasdiTips": {
                "$type": "gov.irs.factgraph.persisters.DollarWrapper",
                "item": "0.00"
            },
            "/familyAndHousehold": {
                "$type": "gov.irs.factgraph.persisters.CollectionWrapper",
                "item": {
                    "items": [
                    ]
                }
            },
            "/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/lastName": {
                "$type": "gov.irs.factgraph.persisters.StringWrapper",
                "item": "Dunn"
            },
            "/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/oasdiWages": {
                "$type": "gov.irs.factgraph.persisters.DollarWrapper",
                "item": "90000.00"
            },
            "/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/combatPay": {
                "$type": "gov.irs.factgraph.persisters.DollarWrapper",
                "item": "0.00"
            },
            "/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/isPrimaryFiler": {
                "$type": "gov.irs.factgraph.persisters.BooleanWrapper",
                "item": true
            },
            "/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/oasdiWithholding": {
                "$type": "gov.irs.factgraph.persisters.DollarWrapper",
                "item": "5000.00"
            },
            "/maritalStatus": {
                "$type": "gov.irs.factgraph.persisters.EnumWrapper",
                "item": {
                    "value": [
                        "single"
                    ],
                    "enumOptionsPath": "/maritalStatusOptions"
                }
            },
            "/receivedDigitalAssets": {
                "$type": "gov.irs.factgraph.persisters.BooleanWrapper",
                "item": false
            },
            "/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/tin": {
                "$type": "gov.irs.factgraph.persisters.TinWrapper",
                "item": {"area":"121","group":"12","serial":"3121"}
            },
            "/wantsStandardDeduction": {
                "$type": "gov.irs.factgraph.persisters.BooleanWrapper",
                "item": true
            },
            "/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/medicareWages": {
                "$type": "gov.irs.factgraph.persisters.DollarWrapper",
                "item": "90000.00"
            },
            "/interestIncome": {
                "$type": "gov.irs.factgraph.persisters.DollarWrapper",
                "item": "0.00"
            },
            "/phone": {
                "$type": "gov.irs.factgraph.persisters.E164Wrapper",
                "item": {
                    "$type": "gov.irs.factgraph.types.UsPhoneNumber",
                    "areaCode": "202",
                    "officeCode": "555",
                    "lineNumber": "0100"
                }
            },
            "/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/isBlind": {
                "$type": "gov.irs.factgraph.persisters.BooleanWrapper",
                "item": false
            },
            "/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/federalWithholding": {
                "$type": "gov.irs.factgraph.persisters.DollarWrapper",
                "item": "30000.00"
            },
            "/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/firstName": {
                "$type": "gov.irs.factgraph.persisters.StringWrapper",
                "item": "Todd"
            },
            "/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/wages": {
                "$type": "gov.irs.factgraph.persisters.DollarWrapper",
                "item": "80000.00"
            },
            "/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/ein": {
                "$type": "gov.irs.factgraph.persisters.StringWrapper",
                "item": "111121121"
            },
            "/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/dateOfBirth": {
                "$type": "gov.irs.factgraph.persisters.DayWrapper",
                "item": {
                    "date": "2000-01-01"
                }
            },
            "/address": {
                "$type": "gov.irs.factgraph.persisters.AddressWrapper",
                "item": {
                    "streetAddress": "736 Jackson Place NW",
                    "city": "Washington",
                    "postalCode": "20503",
                    "stateOrProvence": "DC"
                }
            },
            "/filingStatus": {
                "$type": "gov.irs.factgraph.persisters.EnumWrapper",
                "item": {
                    "value": [
                        "single"
                    ],
                    "enumOptionsPath": "/filingStatusOptions"
                }
            },
            "/filers": {
                "$type": "gov.irs.factgraph.persisters.CollectionWrapper",
                "item": {
                    "items": [
                        "6b1259fd-8cdb-4efe-bcc8-ad40e604c98b"
                    ]
                }
            },
            "/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/filer": {
                "$type": "gov.irs.factgraph.persisters.CollectionItemWrapper",
                "item": {
                    "id": "6b1259fd-8cdb-4efe-bcc8-ad40e604c98b"
                }
            },
            "/formW2s/#3d1946aa-7280-43d4-b5c9-5fde6a6ba28c/allocatedTips": {
                "$type": "gov.irs.factgraph.persisters.DollarWrapper",
                "item": "0.00"
            }
        }
    }""";
}
