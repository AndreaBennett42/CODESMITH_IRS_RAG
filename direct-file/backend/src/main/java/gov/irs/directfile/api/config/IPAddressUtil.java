package gov.irs.directfile.api.config;

import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.lang3.StringUtils;

@SuppressWarnings(value = {"PMD.AvoidUsingHardCodedIP", "PMD.SignatureDeclareThrowsException"})
@Slf4j
public class IPAddressUtil {

    public static String getClientIpAddress(HttpServletRequest request) throws Exception {
        String trueClientIpHeaderValue = request.getHeader(RequestHeaderNames.TRUE_CLIENT_IP);
        if (StringUtils.isNotBlank(trueClientIpHeaderValue)) {
            return trueClientIpHeaderValue.strip();
        }

        String addr = request.getHeader(RequestHeaderNames.X_FORWARDED_FOR);
        if (StringUtils.isBlank(addr)) {
            return request.getRemoteAddr();
        }

        String[] addrs = addr.split(",");
        return getFirstIpAddress(addrs);
    }

    private static String getFirstIpAddress(String[] xffIpAddresses) throws Exception {
        List<String> ipAddresses = Arrays.asList(xffIpAddresses);

        Optional<String> result = ipAddresses.stream()
                .filter(ip -> ip != null && !ip.isEmpty())
                .map(String::strip)
                .findFirst();

        if (result.isPresent()) {
            return result.get();
        } else {
            throw new Exception("No IP address found.");
        }
    }
}
