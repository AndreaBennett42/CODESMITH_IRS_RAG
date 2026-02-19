package gov.irs.directfile.models;

import java.util.ArrayList;
import java.util.UUID;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class FactEvaluationResultTest {
    @Test
    public void removesSpacesFromAllValues() {
        final FactEvaluationResult facts = new FactEvaluationResult();
        facts.put("/test", "  trim     me  ");
        facts.put("/anotherFact", "");
        facts.put("/nullFact", null);
        facts.put("/noSpace", "no spaces doesn't change");
        facts.put("/frontOnly", "     front spaces are a no");
        facts.put("/backOnly", "another test     ");
        facts.put("/middleFix", "name      here");
        assertEquals("trim me", facts.getString("/test"));
        assertEquals("", facts.getString("/anotherFact"));
        assertTrue(facts.getOptional("/nullFact").isEmpty());
        assertEquals("no spaces doesn't change", facts.getString("/noSpace"));
        assertEquals("front spaces are a no", facts.getString("/frontOnly"));
        assertEquals("another test", facts.getString("/backOnly"));
        assertEquals("name here", facts.getString("/middleFix"));
    }

    @Test
    void handlesNullsJustInCase() {
        final FactEvaluationResult facts = new FactEvaluationResult();
        facts.put("/formW2s", null);
        assertTrue(facts.getOptional("/formW2s").isEmpty());
    }

    @Test
    void collectionIndexingPaths() {
        final FactEvaluationResult facts = new FactEvaluationResult();
        final var idList = new ArrayList<UUID>();
        final UUID id = UUID.randomUUID();
        idList.add(id);
        facts.put("/collection", idList);
        facts.put("/collection/#" + id.toString() + "/fact", true);

        // Valid index should return expected values.
        String path = "/collection/[0]/fact";
        var optional = facts.getOptional(path);
        assertTrue(optional.isPresent());
        assertEquals(optional.get().getClass(), Boolean.class);
        assertTrue((boolean) optional.get());
        assertTrue(facts.getBoolean(path));
        assertTrue(facts.getString(path).equals("true"));

        // Inalid index should return safe default values.
        path = "/collection/[1]/fact";
        optional = facts.getOptional(path);
        assertTrue(optional.isEmpty());
        assertFalse(facts.getBoolean(path));
        assertTrue(facts.getString(path).equals(""));
    }
}
