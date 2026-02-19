import { describe, expect, it } from 'vitest';
import { processPopulateResult } from '../processPopulateResult.js';
import { DataImportRootResponseSchema } from '../schema/DataImportServiceResponse.js';

describe(`Data Import Fetch`, () => {
  it(`should parse the response as of december 23`, async () => {
    const rootParse = DataImportRootResponseSchema.parse(JSON.parse(TEST_RESPONSE));
    const populateResult = processPopulateResult(rootParse, `DATA_IMPORT_ABOUT_YOU_BASIC_PLUS_IP_PIN_PLUS_W2`);
    expect(populateResult.data.aboutYouBasic.state).toEqual(`success`);
    expect(populateResult.data.w2s.state).toEqual(`success`);
  });
});

const TEST_RESPONSE = `{
    "data": {
        "aboutYouBasic": {
            "payload": {
                "createdDate": "2024-12-02",
                "dateOfBirth": "1992-02-09",
                "email": "John103743004.Doe103743004103743004@example.com",
                "landlineNumber": null,
                "mobileNumber": "5555553004",
                "firstName": "John103743004",
                "middleInitial": null,
                "lastName": "Doe103743004",
                "streetAddress": "20th St. & Constitution Ave. NW",
                "streetAddressLine2": null,
                "city": "Washington",
                "stateOrProvence": "DC",
                "postalCode": "20560",
                "tags": [
                    "BIOGRAPHICAL"
                ],
                "source": "SADI"
            },
            "createdAt": "2024-12-23 23:12:51.033",
            "state": "success"
        },
        "ipPin": {
            "payload": null,
            "createdAt": null,
            "state": "incomplete"
        },
        "w2s": {
            "payload": {
                "w2s": [
                    {
                        "ein": "111111111",
                        "employersAddress": {
                            "nameLine": "LADYBIRD LLC",
                            "streetAddressLine1": "123 BOLLEYWOOD BOULEVARD",
                            "streetAddressLine2": "",
                            "city": "MILWAUKEE",
                            "state": "DC",
                            "zipCode": "22221"
                        },
                        "controlNumber": "2023w2101",
                        "employeeAddress": {
                            "nameLine": "John Doe II",
                            "streetAddressLine1": "2121 BLACK JACK PLACE",
                            "streetAddressLine2": "",
                            "city": "MILWAUKEE",
                            "state": "VA",
                            "zipCode": "53221"
                        },
                        "wagesTipsOtherCompensation": "16038",
                        "federalIncomeTaxWithheld": "17037",
                        "socialSecurityWages": "20038",
                        "socialSecurityTaxWithheld": "26038",
                        "medicareWagesAndTips": "16038",
                        "medicareTaxWithheld": "15038",
                        "socialSecurityTips": "18038",
                        "allocatedTips": "16038",
                        "dependentCareBenefits": "1038",
                        "nonQualifiedPlans": "1000",
                        "statutoryEmployeeIndicator": true,
                        "thirdPartySickPayIndicator": true,
                        "retirementPlanIndicator": true
                    },
                    {
                        "ein": "111111112",
                        "employersAddress": {
                            "nameLine": "TWINBIRD LLC",
                            "streetAddressLine1": "345 BOLLEYWOOD BOULEVARD",
                            "streetAddressLine2": "SUITE #12",
                            "city": "MILWAUKEE",
                            "state": "DC",
                            "zipCode": "22222"
                        },
                        "controlNumber": "2023w2102",
                        "employeeAddress": {
                            "nameLine": "John Doe II",
                            "streetAddressLine1": "2121 BLACK JACK PLACE",
                            "streetAddressLine2": "",
                            "city": "MILWAUKEE",
                            "state": "VA",
                            "zipCode": "53221"
                        },
                        "wagesTipsOtherCompensation": "16038",
                        "federalIncomeTaxWithheld": "17037",
                        "socialSecurityWages": "20038",
                        "socialSecurityTaxWithheld": "26038",
                        "medicareWagesAndTips": "16038",
                        "medicareTaxWithheld": "15038",
                        "socialSecurityTips": "18038",
                        "allocatedTips": "16038",
                        "dependentCareBenefits": "1038",
                        "nonQualifiedPlans": "1000",
                        "box12s": [
                            {
                                "code": "Z",
                                "amount": "1000"
                            },
                            {
                                "code": "BB",
                                "amount": "1200"
                            },
                            {
                                "code": "DD",
                                "amount": "1100"
                            },
                            {
                                "code": "EE",
                                "amount": "227257"
                            },
                            {
                                "code": "FF",
                                "amount": "1100"
                            }
                        ],
                        "statutoryEmployeeIndicator": false,
                        "thirdPartySickPayIndicator": true,
                        "retirementPlanIndicator": false
                    },
                    {
                        "ein": "111111113",
                        "employersAddress": {
                            "nameLine": "EMPLOYER LLC",
                            "streetAddressLine1": "346 BOLLEYWOOD AVENEW",
                            "streetAddressLine2": "SUITE 12",
                            "city": "LANHAM",
                            "state": "MD",
                            "zipCode": "22223"
                        },
                        "controlNumber": "2023w2103",
                        "employeeAddress": {
                            "nameLine": "John Doe II",
                            "streetAddressLine1": "2121 BLACK JACK PLACE",
                            "streetAddressLine2": "",
                            "city": "MILWAUKEE",
                            "state": "VA",
                            "zipCode": "53221"
                        },
                        "wagesTipsOtherCompensation": "16038",
                        "federalIncomeTaxWithheld": "17037",
                        "socialSecurityWages": "20038",
                        "socialSecurityTaxWithheld": "26038",
                        "medicareWagesAndTips": "16038",
                        "medicareTaxWithheld": "15038",
                        "socialSecurityTips": "18038",
                        "allocatedTips": "16038",
                        "dependentCareBenefits": "1038",
                        "nonQualifiedPlans": "1000",
                        "box12s": [
                            {
                                "code": "Z",
                                "amount": "1000"
                            },
                            {
                                "code": "HH",
                                "amount": "1100"
                            },
                            {
                                "code": "BB",
                                "amount": "1200"
                            }
                        ],
                        "statutoryEmployeeIndicator": true,
                        "thirdPartySickPayIndicator": true,
                        "retirementPlanIndicator": false
                    },
                    {
                        "ein": "111111116",
                        "employersAddress": {
                            "nameLine": "REDROBIN LLC",
                            "streetAddressLine1": "346 BOLLEYWOOD AVENEW",
                            "streetAddressLine2": " APT/75",
                            "city": "LANHAM",
                            "state": "MD",
                            "zipCode": "22223"
                        },
                        "controlNumber": "2023w2106",
                        "employeeAddress": {
                            "nameLine": "John Doe II",
                            "streetAddressLine1": "2121 BLACK JACK PLACE",
                            "streetAddressLine2": "",
                            "city": "MILWAUKEE",
                            "state": "VA",
                            "zipCode": ""
                        },
                        "wagesTipsOtherCompensation": "16038",
                        "federalIncomeTaxWithheld": "17037",
                        "socialSecurityWages": "0",
                        "socialSecurityTaxWithheld": "26038",
                        "medicareWagesAndTips": "0",
                        "medicareTaxWithheld": "15038",
                        "socialSecurityTips": "18038",
                        "allocatedTips": "16038",
                        "dependentCareBenefits": "1038",
                        "nonQualifiedPlans": "1000",
                        "box12s": [
                            {
                                "code": "Z",
                                "amount": "1000"
                            },
                            {
                                "code": "BB",
                                "amount": "1200"
                            },
                            {
                                "code": "HH",
                                "amount": "1100"
                            }
                        ],
                        "statutoryEmployeeIndicator": false,
                        "thirdPartySickPayIndicator": false,
                        "retirementPlanIndicator": true
                    },
                    {
                        "ein": "111111117",
                        "employersAddress": {
                            "nameLine": "ABC LLC",
                            "streetAddressLine1": "346 STREET AVENEW UNIT 12",
                            "streetAddressLine2": "",
                            "city": "LANHAM",
                            "state": "MD",
                            "zipCode": "22223"
                        },
                        "controlNumber": "2023w2107",
                        "employeeAddress": {
                            "nameLine": "John Doe II",
                            "streetAddressLine1": "2121 BLACK JACK PLACE",
                            "streetAddressLine2": "",
                            "city": "MILWAUKEE",
                            "state": "VA",
                            "zipCode": "53221"
                        },
                        "wagesTipsOtherCompensation": "16038",
                        "federalIncomeTaxWithheld": "17037",
                        "socialSecurityWages": "20038",
                        "socialSecurityTaxWithheld": "26038",
                        "medicareWagesAndTips": "16038",
                        "medicareTaxWithheld": "15038",
                        "socialSecurityTips": "18038",
                        "allocatedTips": "16038",
                        "dependentCareBenefits": "1038",
                        "nonQualifiedPlans": "1000",
                        "box12s": [
                            {
                                "code": "Z",
                                "amount": "1000"
                            },
                            {
                                "code": "BB",
                                "amount": "1200"
                            },
                            {
                                "code": "HH",
                                "amount": "1100"
                            }
                        ],
                        "statutoryEmployeeIndicator": false,
                        "thirdPartySickPayIndicator": false,
                        "retirementPlanIndicator": true
                    }
                ],
                "tags": [
                    "W2"
                ],
                "source": "EDP"
            },
            "createdAt": "2024-12-23 23:12:55.53",
            "state": "success"
        },
        "timeSinceCreation": 23275
    }
}
`;
