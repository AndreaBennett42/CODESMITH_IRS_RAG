package gov.irs.directfile.status.domain;

import java.time.LocalDate;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonBackReference;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;
import org.hibernate.annotations.CreationTimestamp;

@Entity(name = "Error")
@Getter
@Setter
@ToString(doNotUseGetters = true)
public class Error {
    @Id
    @Column(name = "meferror_code", length = 25)
    private String mefErrorCode;

    @Column(name = "error_message", columnDefinition = "TEXT")
    private String errorMessage;

    @Column(name = "meferror_category")
    private String mefErrorCategory;

    private String errorCodeTranslationKey;

    @Column(name = "created_at", nullable = false, updatable = false, columnDefinition = "DEFAULT CURRENT_TIMESTAMP")
    @CreationTimestamp
    private LocalDate createdAt;

    @ManyToMany(mappedBy = "errors")
    @JsonBackReference
    private List<Completed> completed;

    public List<Completed> getCompleted() {
        return List.copyOf(completed);
    }

    public void setCompleted(List<Completed> completed) {
        this.completed = List.copyOf(completed);
    }
}
