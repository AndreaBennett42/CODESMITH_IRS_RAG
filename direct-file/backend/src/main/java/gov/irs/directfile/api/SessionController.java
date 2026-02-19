package gov.irs.directfile.api;

import io.swagger.v3.oas.annotations.Operation;
import jakarta.servlet.http.HttpServletRequest;
import lombok.AllArgsConstructor;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import gov.irs.directfile.api.audit.Auditable;
import gov.irs.directfile.api.events.EventId;
import gov.irs.directfile.api.user.UserService;

@RestController
@RequestMapping("${direct-file.api-version}/session")
@AllArgsConstructor
public class SessionController {
    private final UserService userService;

    @Operation()
    @Auditable(event = EventId.KEEP_ALIVE)
    @GetMapping("/keep-alive")
    public ResponseEntity<Void> keepAlive(HttpServletRequest request) {
        userService.getCurrentUserInfo();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        return new ResponseEntity<>(headers, HttpStatus.OK);
    }
}
