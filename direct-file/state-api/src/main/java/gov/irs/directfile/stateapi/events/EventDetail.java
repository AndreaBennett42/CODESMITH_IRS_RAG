package gov.irs.directfile.stateapi.events;

import java.util.HashMap;
import java.util.Map;

import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.StringUtils;

@Getter
@Setter
public class EventDetail {

    private HashMap<String, String> detailMap;

    public EventDetail() {
        detailMap = new HashMap<>();
    }

    public void addDetail(String key, String val) {
        if (StringUtils.isNotEmpty(key) && StringUtils.isNotEmpty(val)) {
            detailMap.put(key, val);
        }
    }

    public Map<String, String> getDetailMap() {
        return detailMap;
    }
}
