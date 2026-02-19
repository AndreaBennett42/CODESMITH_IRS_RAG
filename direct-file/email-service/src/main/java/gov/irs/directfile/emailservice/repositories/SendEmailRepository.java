package gov.irs.directfile.emailservice.repositories;

import java.util.UUID;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import gov.irs.directfile.emailservice.domain.SendEmail;

@Repository
public interface SendEmailRepository extends CrudRepository<SendEmail, UUID> {}
