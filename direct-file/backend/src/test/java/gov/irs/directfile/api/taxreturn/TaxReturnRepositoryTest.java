package gov.irs.directfile.api.taxreturn;

import java.util.*;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.IntNode;
import lombok.SneakyThrows;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.orm.jpa.TestEntityManager;
import org.springframework.boot.test.context.TestConfiguration;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Bean;

import gov.irs.directfile.api.config.identity.IdentityAttributes;
import gov.irs.directfile.api.config.identity.IdentitySupplier;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;
import gov.irs.directfile.api.user.models.User;
import gov.irs.directfile.api.util.base.BaseRepositoryTest;
import gov.irs.directfile.models.FactTypeWithItem;
import gov.irs.directfile.models.encryption.DataEncryptDecrypt;

import static org.assertj.core.api.Assertions.assertThat;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.*;

@ExtendWith(MockitoExtension.class)
class TaxReturnRepositoryTest extends BaseRepositoryTest {
    @TestConfiguration
    public static class TestConfig {
        // For autowiring into TaxReturnEntityListener
        @Bean
        public ObjectMapper getObjectMapper() {
            return new ObjectMapper();
        }
    }

    @MockBean
    private DataEncryptDecrypt dataEncryptDecrypt;

    @MockBean
    private IdentitySupplier mockIdentitySupplier;

    @Autowired
    private TestEntityManager entityManager;

    @Autowired
    private TaxReturnRepository taxReturnRepo;

    @Autowired
    private ObjectMapper objectMapper;

    @BeforeEach
    void configure() {
        doReturn(new IdentityAttributes(UUID.randomUUID(), UUID.randomUUID(), "email@example.com", "123456789"))
                .when(mockIdentitySupplier)
                .get();
    }

    @Test
    void givenUserExists_whenFindByUserId_thenShouldSucceed() throws JsonProcessingException {
        // given user has no returns
        User user = new User(UUID.fromString("738fc2dc-88f9-4b5c-ace9-c602509ba161"));
        user = entityManager.persist(user);

        // when
        List<TaxReturn> taxReturns = taxReturnRepo.findByUserId(user.getId());

        // then
        assertThat(taxReturns.size()).isEqualTo(0);

        // given user has one return
        TaxReturn taxReturn0 = TaxReturn.testObjectFactoryNoId();
        user.addTaxReturn(taxReturn0);

        taxReturn0 = entityManager.persist(taxReturn0);

        // when
        taxReturns = taxReturnRepo.findByUserId(user.getId());

        // then
        assertThat(taxReturns.size()).isEqualTo(1);

        // given user has two returns (this depends on results being ordered by tax_year descending)
        TaxReturn taxReturn1 = TaxReturn.testObjectFactoryNoId();
        taxReturn1.setTaxYear(taxReturn0.getTaxYear() - 1);
        user.addTaxReturn(taxReturn1);

        taxReturn1 = entityManager.persist(taxReturn1);

        // when
        taxReturns = taxReturnRepo.findByUserId(user.getId());

        // then
        assertThat(taxReturns.size()).isEqualTo(2);
        assertThat(taxReturns.get(0).getId()).isEqualTo(taxReturn0.getId());
        assertThat(taxReturns.get(1).getId()).isEqualTo(taxReturn1.getId());
    }

    @Test
    void givenTaxReturnDoesNotExist_whenFindByUserIdAndTaxYear_thenShouldFail() {
        // given

        // when
        Optional<TaxReturn> result = taxReturnRepo.findByUserIdAndTaxYear(null, 0);

        // then
        assertTrue(result.isEmpty());
    }

    @SneakyThrows
    @Test
    void givenTaxReturnExists_whenFindByUserIdAndTaxYear_thenShouldSucceed() {
        // given
        User user = new User(UUID.fromString("738fc2dc-88f9-4b5c-ace9-c602509ba161"));
        user = entityManager.persist(user);
        TaxReturn taxReturn = TaxReturn.testObjectFactoryNoId();
        user.addTaxReturn(taxReturn);

        taxReturn = entityManager.persist(taxReturn);

        // when
        Optional<TaxReturn> result = taxReturnRepo.findByUserIdAndTaxYear(user.getId(), taxReturn.getTaxYear());

        // then
        assertTrue(result.isPresent());
        assertThat(result.get()).isEqualTo(taxReturn);
    }

    @Test
    void givenTaxReturnDoesNotExist_whenFindByIdAndUserId_thenShouldFail() {
        // given

        // when
        Optional<TaxReturn> result = taxReturnRepo.findByIdAndUserId(null, null);

        // then
        assertTrue(result.isEmpty());
    }

    @SneakyThrows
    @Test
    void givenTaxReturnExists_whenFindByIdAndUserId_thenShouldSucceed() {
        // given
        User user = new User(UUID.fromString("738fc2dc-88f9-4b5c-ace9-c602509ba161"));
        user = entityManager.persist(user);
        TaxReturn taxReturn = TaxReturn.testObjectFactoryNoId();

        user.addTaxReturn(taxReturn);
        taxReturn = entityManager.persist(taxReturn);

        // when
        Optional<TaxReturn> result = taxReturnRepo.findByIdAndUserId(taxReturn.getId(), user.getId());

        // then
        assertTrue(result.isPresent());
        assertThat(result.get()).isEqualTo(taxReturn);
    }

    @SneakyThrows
    @Test
    void givenTaxReturnIsCreated_whenCreateAndFind_thenFactsEncryptionConverterInvoked() {
        // given user has no returns
        User user = new User(UUID.fromString("738fc2dd-88f9-4b5c-ace9-c602509ba161"));
        user = entityManager.persist(user);

        // given user has one return
        TaxReturn taxReturn = TaxReturn.testObjectFactoryNoId();
        taxReturn.setFacts(Map.of("testA", new FactTypeWithItem("typeA", new IntNode(24))));
        byte[] factsBytes = objectMapper.writeValueAsBytes(taxReturn.getFacts());
        when(dataEncryptDecrypt.encrypt(eq(factsBytes), anyMap())).thenReturn(factsBytes);
        when(dataEncryptDecrypt.decrypt(factsBytes)).thenReturn(factsBytes);
        user.addTaxReturn(taxReturn);
        entityManager.persist(taxReturn);

        // when
        List<TaxReturn> taxReturns = taxReturnRepo.findByUserId(user.getId());

        // then
        assertThat(taxReturns.size()).isEqualTo(1);
    }

    @SneakyThrows
    @Test
    void givenTaxReturnExist_whenFindByTaxReturnIds_thenShouldBePresent() {
        // given
        User user = new User(UUID.fromString("738fc2dc-88f9-4b5c-ace9-c602509ba161"));
        user = entityManager.persist(user);
        TaxReturn taxReturn = TaxReturn.testObjectFactoryNoId();

        user.addTaxReturn(taxReturn);
        taxReturn = entityManager.persist(taxReturn);

        // when
        List<TaxReturn> result = taxReturnRepo.findAllByTaxReturnIds(List.of(taxReturn.getId()));

        // then
        assertTrue(result.stream().findFirst().isPresent());
        assertThat(result).isEqualTo(List.of(taxReturn));
        assertThat(result.stream().findFirst().get()).isEqualTo(taxReturn);
    }

    @SneakyThrows
    @Test
    void givenManyTaxReturnExists_whenFindByTaxReturnIds_thenAllShouldBePresent() {
        // given
        List<UUID> trIDs = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            User user = new User(UUID.fromString("738fc2dc-88f9-4b5c-ace9-c602509ea16" + i));
            user = entityManager.persist(user);
            TaxReturn taxReturn = TaxReturn.testObjectFactoryNoId();

            user.addTaxReturn(taxReturn);
            taxReturn = entityManager.persist(taxReturn);
            trIDs.add(taxReturn.getId());
        }

        // when
        List<TaxReturn> result = taxReturnRepo.findAllByTaxReturnIds(trIDs);

        // then
        assertEquals(result.size(), 5);
        trIDs.forEach(trID -> {
            assertTrue(result.contains(taxReturnRepo.findById(trID).get()));
        });
    }
}
