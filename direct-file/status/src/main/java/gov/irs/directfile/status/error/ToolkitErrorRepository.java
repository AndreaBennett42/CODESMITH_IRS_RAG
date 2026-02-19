package gov.irs.directfile.status.error;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import gov.irs.directfile.status.domain.ToolkitError;

@Repository
public interface ToolkitErrorRepository extends CrudRepository<ToolkitError, String> {}
