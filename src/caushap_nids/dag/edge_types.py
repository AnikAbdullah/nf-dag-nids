from enum import Enum


class EdgeType(Enum):
    P = "protocol"   # Required by RFC 3954/7011 — highest trust
    M = "mitre"      # Implied by MITRE ATT&CK technique-to-feature mappings
    D = "data"       # Supported by PC/NOTEARS on benign training data
    U = "uncertain"  # Flagged for review; kept if >= 2 methods agree

    @property
    def priority(self) -> int:
        return {"P": 3, "M": 2, "D": 1, "U": 0}[self.name]
