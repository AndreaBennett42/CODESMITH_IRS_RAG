package gov.irs.directfile.status.error;

import org.springframework.data.repository.CrudRepository;

import gov.irs.directfile.status.domain.Error;

public interface ErrorRepository extends CrudRepository<Error, String> {}
