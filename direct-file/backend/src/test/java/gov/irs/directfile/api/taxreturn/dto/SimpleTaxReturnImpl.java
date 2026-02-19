package gov.irs.directfile.api.taxreturn.dto;

import java.util.Date;
import java.util.UUID;

import lombok.Setter;

import gov.irs.directfile.api.taxreturn.SimpleTaxReturnProjection;
import gov.irs.directfile.api.taxreturn.models.TaxReturn;

@Setter
public class SimpleTaxReturnImpl implements SimpleTaxReturnProjection {
    private Date createdAt;
    private Date updatedAt;
    private Date submitTime;
    private UUID id;
    private int taxYear;

    public SimpleTaxReturnImpl() {}

    public SimpleTaxReturnImpl(Date createdAt, Date updatedAt, Date submitTime, UUID id, int taxYear) {
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
        this.submitTime = submitTime;
        this.id = id;
        this.taxYear = taxYear;
    }

    @Override
    public Date getCreatedAt() {
        return this.createdAt;
    }

    @Override
    public UUID getId() {
        return this.id;
    }

    @Override
    public Date getUpdatedAt() {
        return this.updatedAt;
    }

    @Override
    public Date getSubmitTime() {
        return this.submitTime;
    }

    @Override
    public int getTaxYear() {
        return this.taxYear;
    }

    public static SimpleTaxReturnProjection fromTaxReturn(TaxReturn taxReturn) {
        return new SimpleTaxReturnImpl(
                taxReturn.getCreatedAt(),
                taxReturn.getUpdatedAt(),
                taxReturn.getSubmitTime(),
                taxReturn.getId(),
                taxReturn.getTaxYear());
    }
}
