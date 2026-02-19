package gov.irs.directfile.api.authentication;

import java.util.Collection;
import java.util.UUID;

import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

@SuppressWarnings("PMD.ReturnEmptyCollectionRatherThanNull")
public record SMUserDetailsPrincipal(UUID id, UUID externalId, String email, String tin) implements UserDetails {
    public SMUserDetailsPrincipal(SMUserDetailsProperties properties) {
        this(properties.id(), properties.externalId(), properties.email(), properties.tin());
    }

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return null;
    }

    @Override
    public String getUsername() {
        return email;
    }

    @Override
    public String getPassword() {
        return null;
    }

    @Override
    public boolean isAccountNonExpired() {
        return true;
    }

    @Override
    public boolean isAccountNonLocked() {
        return true;
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return true;
    }

    @Override
    public boolean isEnabled() {
        return true;
    }
}
