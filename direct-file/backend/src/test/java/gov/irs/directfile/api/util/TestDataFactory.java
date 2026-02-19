package gov.irs.directfile.api.util;

import java.util.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.*;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import gov.irs.directfile.api.taxreturn.TaxReturnRepository;
import gov.irs.directfile.api.taxreturn.TaxReturnSubmissionRepository;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.taxreturn.models.TaxReturnSubmission;
import gov.irs.directfile.api.user.UserRepository;
import gov.irs.directfile.api.user.models.User;
import gov.irs.directfile.models.FactTypeWithItem;
import gov.irs.directfile.models.message.event.SubmissionEventTypeEnum;

@Component
@Transactional
public class TestDataFactory {
    @Autowired
    private UserRepository userRepository;

    @Autowired
    private TaxReturnRepository taxReturnRepository;

    @Autowired
    private TaxReturnSubmissionRepository taxReturnSubmissionRepository;

    @Autowired
    ObjectMapper objectMapper;

    public User createUserFromTestUser(SecurityTestConfiguration.TestUserProperties testUserProperties) {
        User dbUser = new User(testUserProperties.getExternalId());
        return userRepository.save(dbUser);
    }

    public TaxReturn addTaxReturnToUserByUserExternalId(final UUID userExternalId) {
        return addTaxReturnToUserByUserExternalId(userExternalId, new HashMap<>());
    }

    public TaxReturn addTaxReturnToUserByUserExternalId(
            final UUID userExternalId, final Map<String, FactTypeWithItem> facts) {
        return _addTaxReturnToUserByUserExternalId(userExternalId, facts);
    }

    public TaxReturn _addTaxReturnToUserByUserExternalId(
            final UUID userExternalId, final Map<String, FactTypeWithItem> facts) {
        User user = userRepository.findByExternalId(userExternalId).get();
        TaxReturn taxReturn = new TaxReturn();
        taxReturn.setTaxYear(2024);
        taxReturn.addOwner(user);
        taxReturn.setFacts(facts);
        return taxReturnRepository.save(taxReturn);
    }

    public TaxReturnSubmission addAcceptedTaxReturnSubmissionToTaxReturn(TaxReturn taxReturn) {
        TaxReturnSubmission taxReturnSubmission = taxReturn.addTaxReturnSubmission();
        taxReturnSubmission.addSubmissionEvent(SubmissionEventTypeEnum.SUBMITTED);
        taxReturnSubmission.addSubmissionEvent(SubmissionEventTypeEnum.ACCEPTED);
        taxReturn = taxReturnRepository.save(taxReturn);
        return taxReturnSubmissionRepository
                .findLatestTaxReturnSubmissionByTaxReturnId(taxReturn.getId())
                .get();
    }

    public HashMap<String, FactTypeWithItem> auditedFacts() throws JsonProcessingException {
        ArrayList<JsonNode> filersArray = new ArrayList<>();
        JsonNodeFactory jsonNodeFactory = new JsonNodeFactory(false);
        filersArray.add(jsonNodeFactory.textNode("6b1259fd-8cdb-4efe-bcc8-ad40e604c98b"));
        filersArray.add(jsonNodeFactory.textNode(UUID.randomUUID().toString()));
        LinkedHashMap<String, JsonNode> tinMap = new LinkedHashMap<>();
        tinMap.put("area", new TextNode("121"));
        tinMap.put("group", new TextNode("12"));
        tinMap.put("serial", new TextNode("3121"));
        ObjectNode emailNode = JsonNodeFactory.instance.objectNode();
        objectMapper = new ObjectMapper();
        emailNode.put("email", "user@example.com");

        HashMap<String, FactTypeWithItem> facts = new HashMap<>(Map.of(
                "/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/tin",
                new FactTypeWithItem(
                        "gov.irs.factgraph.persisters.TinWrapper", new ObjectNode(new JsonNodeFactory(false), tinMap)),
                "/filers",
                new FactTypeWithItem(
                        "gov.irs.factgraph.persisters.CollectionWrapper",
                        new ObjectNode(
                                new JsonNodeFactory(false),
                                Map.of("items", new ArrayNode(new JsonNodeFactory(false), filersArray)))),
                "/filers/#6b1259fd-8cdb-4efe-bcc8-ad40e604c98b/isPrimaryFiler",
                new FactTypeWithItem("gov.irs.factgraph.persisters.BooleanWrapper", BooleanNode.getTrue()),
                "/email",
                new FactTypeWithItem("gov.irs.factgraph.persisters.EmailAddressWrapper", emailNode),
                "/filerResidenceAndIncomeState",
                new FactTypeWithItem(
                        "gov.irs.factgraph.persisters.EnumWrapper",
                        objectMapper.readTree("{\n" + "               \"value\": [\n"
                                + "                 \"wa\"\n"
                                + "               ],\n"
                                + "               \"enumOptionsPath\": \"/scopedStateOptions\"\n"
                                + "             }"))));
        return facts;
    }

    public Map<String, FactTypeWithItem> getMinimalFactsToProvideTin(final String userTin)
            throws JsonProcessingException {
        String tinArea = userTin.substring(0, 3);
        String tinGroup = userTin.substring(3, 5);
        String tinSerial = userTin.substring(5, 9);
        return Map.of(
                "/filers",
                new FactTypeWithItem(
                        "gov.irs.factgraph.persisters.CollectionWrapper",
                        objectMapper.readTree("{\"items\":[\"00000000-0000-0000-0000-000000000000\"]}")),
                "/filers/#00000000-0000-0000-0000-000000000000/isPrimaryFiler",
                new FactTypeWithItem("gov.irs.factgraph.persisters.BooleanWrapper", BooleanNode.TRUE),
                "/filers/#00000000-0000-0000-0000-000000000000/tin",
                new FactTypeWithItem(
                        "gov.irs.factgraph.persisters.TinWrapper",
                        objectMapper.readTree(String.format(
                                "{\"area\":\"%s\",\"group\":\"%s\",\"serial\":\"%s\"}",
                                tinArea, tinGroup, tinSerial))));
    }
}
