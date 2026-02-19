package gov.irs.directfile.stateapi.utils;

import jakarta.servlet.http.HttpServletRequest;
import lombok.experimental.UtilityClass;

@UtilityClass
public class IPAddress {
    private static String X_FORWARDED_FOR = "X-Forwarded-For";

    public String getClientIpAddress(HttpServletRequest request) {
        String addr = request.getHeader(X_FORWARDED_FOR);
        if (addr == null || addr.isEmpty()) {
            addr = request.getRemoteAddr();
        } else {
            String[] addrs = addr.split(",");
            addr = addrs[0];
        }
        return addr;
    }
}
