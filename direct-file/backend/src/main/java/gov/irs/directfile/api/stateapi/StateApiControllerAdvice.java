package gov.irs.directfile.api.stateapi;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import gov.irs.directfile.api.errors.InvalidDataException;
import gov.irs.directfile.api.errors.NonexistentDataException;
import gov.irs.directfile.api.stateapi.domain.CreateAuthorizationCodeResponse;
import gov.irs.directfile.dto.AuthCodeResponse;
import gov.irs.directfile.error.StateApiErrorCode;

@RestControllerAdvice(assignableTypes = {StateApiController.class})
@Slf4j
public class StateApiControllerAdvice {

    @ExceptionHandler(NonexistentDataException.class)
    @ResponseStatus(HttpStatus.NOT_FOUND)
    public CreateAuthorizationCodeResponse handleNonExistentData(NonexistentDataException ex) {
        return CreateAuthorizationCodeResponse.builder().error(ex.getMessage()).build();
    }

    @ExceptionHandler(InvalidDataException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    public CreateAuthorizationCodeResponse handleInvalidData(InvalidDataException ex) {
        return CreateAuthorizationCodeResponse.builder().error(ex.getMessage()).build();
    }

    @ExceptionHandler(WebClientResponseException.class)
    // status code depends on service response
    // TO DO: Update createAuthorizationCode() (maybe other methods also) in StateApi (the one out of backend) to
    // customize exceptions
    public ResponseEntity<CreateAuthorizationCodeResponse> handleFailedServiceRequest(WebClientResponseException ex) {
        try {
            // See if the response is an AuthCodeResponse
            AuthCodeResponse response = ex.getResponseBodyAs(AuthCodeResponse.class);
            if (response != null && response.getErrorCode() != null) {
                StateApiErrorCode errorCode = response.getErrorCode();
                HttpStatus status = errorCode.getHttpStatus();
                return new ResponseEntity<>(
                        CreateAuthorizationCodeResponse.builder()
                                .error(response.getErrorCode().name())
                                .build(),
                        status);
            }
        } catch (IllegalStateException ignored) {
        }

        return new ResponseEntity<>(
                CreateAuthorizationCodeResponse.builder()
                        .error("Unsuccessful request")
                        .build(),
                ex.getStatusCode());
    }
}
