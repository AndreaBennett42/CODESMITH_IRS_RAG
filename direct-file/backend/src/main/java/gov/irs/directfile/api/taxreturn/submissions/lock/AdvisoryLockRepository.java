package gov.irs.directfile.api.taxreturn.submissions.lock;

import java.util.UUID;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import gov.irs.directfile.api.taxreturn.models.TaxReturn;

// While ZooKeeper or Redis may be better choices for distributed lock management,
// given that we are using PostgreSQL, we opt for the built-in pg advisory lock.
// The pg advisory lock utilizes integer instead of string for minimal performance overhead.
// When multiple threads or pods are connecting to the same PostgreSQL instance and attempting to acquire an advisory
// lock using pg_try_advisory_lock, the lock is indeed shared and visible across all threads or pods. If one thread
// successfully acquires the lock, subsequent attempts by other threads (from same pod or other pods) to acquire the
// same lock will fail (i.e., pg_try_advisory_lock will return false), indicating that the lock acquisition failed.
// Ref: https://medium.com/inspiredbrilliance/a-practical-guide-to-using-advisory-locks-in-your-application-7f0e7908d7e9
public interface AdvisoryLockRepository extends JpaRepository<TaxReturn, UUID> {
    @Query(value = "SELECT pg_try_advisory_lock(:lockId)", nativeQuery = true)
    boolean acquireLock(int lockId);

    @Query(value = "SELECT pg_advisory_lock(:lockId)", nativeQuery = true)
    void acquireLockBlocking(int lockId);

    @Query(value = "SELECT pg_advisory_unlock(:lockId)", nativeQuery = true)
    boolean releaseLock(int lockId);
}
