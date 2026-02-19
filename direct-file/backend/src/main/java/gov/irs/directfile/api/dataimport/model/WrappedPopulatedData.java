package gov.irs.directfile.api.dataimport.model;

import java.text.ParseException;
import java.util.*;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.JsonNodeFactory;
import com.fasterxml.jackson.databind.node.ObjectNode;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Getter
@RequiredArgsConstructor
@SuppressWarnings({"PMD.SimpleDateFormatNeedsLocale", "PMD.AvoidDuplicateLiterals", "PMD.LiteralsFirstInComparisons"})
public class WrappedPopulatedData {

    private final Data data;

    static final String EMPTY_JSON = "{}"; /* e.g. for RAW_DATA which cannot be empty or null */

    @Getter
    @RequiredArgsConstructor
    public static class Data {
        private final WrappedPopulatedDataNode aboutYouBasic;
        private final WrappedPopulatedDataNode ipPin;
        private final WrappedPopulatedDataNode w2s;
        private final WrappedPopulatedDataNode f1099Ints;
        private final WrappedPopulatedDataNode f1095a;

        private final long timeSinceCreation;
    }

    @JsonCreator
    public WrappedPopulatedData(JsonNode _data) throws ParseException {
        // this is used to deserialize mock json files for the MockDataImportService
        // there is some differences between how deserialization of the mock json files occurs for Docker and local
        // the below check handles this
        JsonNode data = _data;
        if (!_data.get("data").isEmpty() || !_data.get("data").toString().isBlank()) {
            data = _data.get("data");
        }

        JsonNode aboutYouBasicData = data.get("aboutYouBasic");
        JsonNode ipPinData = data.get("ipPin");
        JsonNode w2sData = data.get("w2s");
        JsonNode f1099IntsData = data.get("f1099Ints");
        JsonNode f1095aData = data.get(
                "f1095a"); // the key in the mocks is f1095A to mimic the GET /populate response, but in the response
        // from the Lambda it is f1095As
        WrappedPopulatedDataNode aboutYouBasic = new WrappedPopulatedDataNode(
                aboutYouBasicData.get("payload"),
                aboutYouBasicData.get("createdAt").textValue(),
                aboutYouBasicData.get("state").asText());
        WrappedPopulatedDataNode ipPin = new WrappedPopulatedDataNode(
                ipPinData.get("payload"),
                ipPinData.get("createdAt").textValue(),
                ipPinData.get("state").asText());
        WrappedPopulatedDataNode w2s = new WrappedPopulatedDataNode(
                w2sData.get("payload"),
                w2sData.get("createdAt").textValue(),
                w2sData.get("state").asText());
        // f1099Ints is optional
        WrappedPopulatedDataNode f1099Ints = null;
        if (f1099IntsData != null) {
            f1099Ints = new WrappedPopulatedDataNode(
                    f1099IntsData.get("payload"),
                    f1099IntsData.get("createdAt").textValue(),
                    f1099IntsData.get("state").asText());
        } else {
            f1099Ints = new WrappedPopulatedDataNode();
        }
        WrappedPopulatedDataNode f1095a = null;
        if (f1095aData != null) {
            f1095a = new WrappedPopulatedDataNode(
                    f1095aData.get("payload"),
                    f1095aData.get("createdAt").textValue(),
                    f1095aData.get("state").asText());
        } else {
            f1095a = new WrappedPopulatedDataNode();
        }

        long timeSinceCreation = data.get("timeSinceCreation").asInt();
        this.data = new Data(aboutYouBasic, ipPin, w2s, f1099Ints, f1095a, timeSinceCreation);
    }

    public static WrappedPopulatedData from(List<PopulatedData> listData, Date taxReturnCreatedAt) {
        WrappedPopulatedDataNode aboutYouBasic = new WrappedPopulatedDataNode();
        WrappedPopulatedDataNode ipPin = new WrappedPopulatedDataNode();
        WrappedPopulatedDataNode w2s = new WrappedPopulatedDataNode();
        WrappedPopulatedDataNode f1099Ints = new WrappedPopulatedDataNode();
        WrappedPopulatedDataNode f1095a = new WrappedPopulatedDataNode();

        for (PopulatedData data : listData) {
            switch (data.getSource()) {
                case "SADI":
                    handleAboutYou(aboutYouBasic, data);
                    break;
                case "IPPIN":
                    handleIPPin(ipPin, data);
                    break;
                case "W2":
                    handleW2(w2s, data.getData(), data.getCreatedAt().toString());
                    break;
                case "FORM_1099_INT":
                    handle1099Int(
                            f1099Ints,
                            data.getData().get("f1099Ints"),
                            data.getCreatedAt().toString());
                    break;
                case "FORM_1095_A":
                    handle1095A(
                            f1095a,
                            data.getData().get("f1095As"),
                            data.getCreatedAt().toString());
                    break;
                default:
            }
        }

        return new WrappedPopulatedData(new Data(
                aboutYouBasic, ipPin, w2s, f1099Ints, f1095a, calculateTimeSinceCreationInMs(taxReturnCreatedAt)));
    }

    private static void handleAboutYou(WrappedPopulatedDataNode aboutYouBasic, PopulatedData data) {
        aboutYouBasic.setPayload(data.getData());
        aboutYouBasic.setCreatedAt(data.getCreatedAt().toString());
        aboutYouBasic.setState(WrappedPopulatedDataNodeState.SUCCESS.getState());
    }

    private static void handleIPPin(WrappedPopulatedDataNode ipPin, PopulatedData data) {
        ipPin.setPayload(data.getData());
        ipPin.setCreatedAt(data.getCreatedAt().toString());
        ipPin.setState(WrappedPopulatedDataNodeState.SUCCESS.getState());
    }

    private static void handleW2(WrappedPopulatedDataNode w2s, JsonNode payload, String createdAt) {
        w2s.setPayload(payload);
        w2s.setCreatedAt(createdAt);
        w2s.setState(WrappedPopulatedDataNodeState.SUCCESS.getState());
    }

    private static void handle1099Int(WrappedPopulatedDataNode f1099Ints, JsonNode payload, String createdAt) {
        f1099Ints.setPayload(payload);
        f1099Ints.setCreatedAt(createdAt);
        f1099Ints.setState(WrappedPopulatedDataNodeState.SUCCESS.getState());
    }

    private static void handle1095A(WrappedPopulatedDataNode f1095a, JsonNode payload, String createdAt) {
        JsonNode responseNode = new ObjectNode(new JsonNodeFactory(false));
        boolean has1095A =
                !payload.isNull() && !payload.isEmpty() && !payload.asText().equals(EMPTY_JSON);
        ((ObjectNode) responseNode).put("has1095A", has1095A);
        f1095a.setPayload(responseNode);
        f1095a.setCreatedAt(createdAt);
        f1095a.setState(WrappedPopulatedDataNodeState.SUCCESS.getState());
    }

    private static long calculateTimeSinceCreationInMs(Date taxReturnCreatedAt) {
        return Calendar.getInstance().getTime().getTime() - taxReturnCreatedAt.getTime();
    }
}
