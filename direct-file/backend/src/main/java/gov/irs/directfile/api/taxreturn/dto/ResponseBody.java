package gov.irs.directfile.api.taxreturn.dto;

import java.util.*;

import lombok.Getter;
import lombok.Setter;

import gov.irs.directfile.api.dataimport.model.WrappedPopulatedData;
import gov.irs.directfile.models.FactTypeWithItem;

@Getter
@Setter
public class ResponseBody {
    private UUID id;
    private Date createdAt;
    private int taxYear;
    private Map<String, FactTypeWithItem> facts;
    private String store;
    private List<TaxReturnSubmissionResponseBody> taxReturnSubmissions;
    private Boolean isEditable;
    private WrappedPopulatedData.Data populatedData;
    private String dataImportBehavior;
    private Boolean surveyOptIn;

    public static final String docsExampleObject =
            """
    {
        "id": "f4b14d1f-fc8b-40e2-9317-deaf69d17f65",
        "createdAt": "2024-10-28 15:55:35.285",
        "taxYear": 2023,
        "dataImportBehavior": 3,
        "surveyOptIn": true,
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
                "item": "Lake"
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
                    "options": [
                        "married",
                        "divorced",
                        "single",
                        "widowed"
                    ],
                    "enumId": "maritalStatus"
                }
            },
            "/receivedDigitalAssets": {
                "$type": "gov.irs.factgraph.persisters.BooleanWrapper",
                "item": false
            },
            "/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/tin": {
                "$type": "gov.irs.factgraph.persisters.StringWrapper",
                "item": "121123121"
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
                    "options": [
                        "single",
                        "qualifiedSurvivingSpouse",
                        "headOfHousehold",
                        "marriedFilingSeparately",
                        "marriedFilingJointly"
                    ],
                    "enumId": "filingStatus"
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
        },
        "taxReturnSubmissions": [
             {
                 "id": "2d59a07d-57ef-4392-8196-48ac29dce023",
                 "createdAt": "2023-10-26 15:04:47.197",
                 "submitUserId": "6b1259fd-8cdb-4efe-bcc8-ad40e604c98b",
                 "submissionId": "12345620230215000001",
                 "submissionReceivedAt": "2023-10-26 16:01:34.221"
             },
             {
                 "id": "0ac15058-9352-49f8-9b84-5e3faed41676",
                 "createdAt": "2023-10-24 18:24:17.109",
                 "submitUserId": "6b1259fd-8cdb-4efe-bcc8-ad40e604c98b",
                 "submissionId": "55555620230215000001"m
                 "submissionReceivedAt": "2023-10-26 16:01:34.221"
             }
         ]
    }""";

    public static final String docsExampleList =
            """
    [
        {
            "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "createdAt": "2024-10-28 15:55:35.285",
            "taxYear": 2022,
            "surveyOptIn": null,
            "facts": {
                "/filingStatus": {
                    "$type": "gov.irs.factgraph.persisters.EnumWrapper",
                    "item": {
                        "value": [
                            "single"
                        ],
                        "options": [
                            "single",
                            "qualifiedSurvivingSpouse",
                            "headOfHousehold",
                            "marriedFilingSeparately",
                            "marriedFilingJointly"
                        ],
                        "enumId": "filingStatus"
                    }
                }
            },
        },
        """
                    + docsExampleObject
                    + """
    ]""";
}
