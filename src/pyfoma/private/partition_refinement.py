#!/usr/bin/env python

class PartitionRefinement:

    """Basic partition refinement using dicts. A pared down version of D. Eppstein's
       implementation. https://www.ics.uci.edu/~eppstein/PADS/PartitionRefinement.py"""

    def __init__(self, S):
        """Create a new partition refinement data structure for the given
        items.  Initially, all items belong to the same subset.
        """
        self.sets = {id(s):s for s in S}
        self.partition = {x:s for s in S for x in s}

    def refine(self, S):
        """Refine each set A in the partition to the two sets
        A & S, A - S.  Return a list of pairs (A & S, A - S)
        for each changed set.  Within each pair, A & S will be
        a newly created set, while A - S will be a modified
        version of an existing set in the partition.
        Not a generator because we need to perform the partition
        even if the caller doesn't iterate through the results.
        """
        hit = {}
        output = []
        for x in S:
            if x in self.partition:
                Ax = self.partition[x]
                hit.setdefault(id(Ax), set()).add(x)
        for A, AS in hit.items():
            A = self.sets[A]
            if AS != A:
                self.sets[id(AS)] = AS
                for x in AS:
                    self.partition[x] = AS
                A -= AS
                output.append((id(AS), id(A)))
        return output

    def astuples(self):
        """Get current partitioning and convert to set of tuples."""
        return {tuple(s) for s in self.sets.values()}
