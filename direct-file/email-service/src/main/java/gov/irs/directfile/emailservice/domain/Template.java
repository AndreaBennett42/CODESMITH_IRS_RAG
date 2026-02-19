package gov.irs.directfile.emailservice.domain;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
@NoArgsConstructor
public class Template {
    private String subject;
    private String body;

    // Is this template considered dynamic (e.g., a template that should be passed through
    // a template language along with a context to create a dynamic result)?
    private boolean dynamic;
}
