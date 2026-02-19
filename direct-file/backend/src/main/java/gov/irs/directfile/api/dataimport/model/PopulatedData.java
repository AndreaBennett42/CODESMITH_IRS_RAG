package gov.irs.directfile.api.dataimport.model;

import java.util.Date;
import java.util.UUID;

import com.fasterxml.jackson.databind.JsonNode;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EntityListeners;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Transient;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

@Getter
@Entity
@Table(name = "populated_data")
@EntityListeners({PopulatedDataEntityListener.class})
public class PopulatedData {
    @Id
    @GeneratedValue(generator = "UUID4")
    @Column(nullable = false, updatable = false)
    private UUID id;

    @Setter
    @Column(name = "taxreturn_id", nullable = false)
    private UUID taxReturnId;

    @Setter
    @Column(name = "source", nullable = false)
    private String source;

    @Setter
    @Column(name = "tags", nullable = false)
    private String tags;

    @Setter
    @Column(name = "data", nullable = false)
    private String dataCipherText;

    @Setter
    @Transient
    private JsonNode data;

    @Setter
    @Column(name = "raw_data", nullable = false)
    private String rawDataCipherText;

    @Setter
    @Transient
    private JsonNode rawResponseData;

    @Setter
    @Column(name = "created_at", nullable = false, updatable = false, columnDefinition = "DEFAULT CURRENT_TIMESTAMP")
    @CreationTimestamp
    private Date createdAt;

    public static final String docsExampleObject =
            """
                     {
                         "timeSinceCreation": 10000,
                         "aboutYouBasic": {
                           "state": "success",
                           "createdAt": "2024-11-27T00:00:00Z",
                           "payload": {
                             "source": "SADI",
                             "tags": ["BIOGRAPHICAL"],
                             "createdDate": "2024-01-01",
                             "dateOfBirth": "1980-08-02",
                             "email": "Homer.Simpson@test.email",
                             "mobileNumber": "+12223334444",
                             "landlineNumber": null,
                             "firstName": "Lisa",
                             "middleInitial": "",
                             "lastName": "Simpson",
                             "streetAddress": "123 Sesame St",
                             "streetAddressLine2": null,
                             "city": "Springfield",
                             "stateOrProvence": "TN",
                             "postalCode": "37172"
                           }
                         },
                         "ipPin": {
                           "state": "success",
                           "createdAt": "2024-11-27T00:00:00Z",
                           "payload": {
                             "source": "IPPIN",
                             "tags": ["BIOGRAPHICAL"],
                             "hasIpPin": false
                           }
                         },
                         "w2s": {
                           "state": "success",
                           "createdAt": "2024-11-27T00:00:00Z",
                           "payload": [
                             {
                               "id": "1a657d0c-b676-4bbf-9d18-d9ecb8547d8d",
                               "source": "EDP",
                               "tags": ["W2"],
                               "ein": "001234567",
                               "firstName": "Lisa",
                               "employerAddress": {
                                 "name": "Goods and Stuff",
                                 "nameLine2": "",
                                 "streetAddress": "7588 PEACH TREE ST",
                                 "streetAddressLine2": "",
                                 "city": "SPRINGFIELD",
                                 "stateOrProvence": "TN",
                                 "postalCode": "37172",
                                 "country": "USA"
                               },
                               "controlNumber": "000011 R#/123",
                               "employeeAddress": null,
                               "wagesTipsOtherCompensation": "20000",
                               "federalIncomeTaxWithheld": "2000",
                               "socialSecurityWages": "20000",
                               "socialSecurityTaxWithheld": "1240",
                               "medicareWagesAndTips": "20000",
                               "medicareTaxWithheld": "290",
                               "socialSecurityTips": "",
                               "allocatedTips": "",
                               "dependentCareBenefits": "",
                               "box12s": [],
                               "statutoryEmployeeIndicator": false,
                               "thirdPartySickPayIndicator": false,
                               "retirementPlanIndicator": false,
                               "nonQualifiedPlans": ""
                             }
                           ]
                         }
                       }
                     }
            """;
}
