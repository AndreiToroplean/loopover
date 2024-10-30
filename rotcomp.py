import random
from itertools import count


class RotComp(list):
    """Represents a sequence (or composition) of Rot objects. Behaves like a list, with additional methods to
    manipulate it while maintaining its value.

    Glossary:
        Group: (not linked to group theory) Set of Rots that form a chain of dependencies. Ie for each two Rots in the
        group, there is a path of Rots that share one index, to link them.
        Cycle: (not formally defined in relation to graph theory) Chain of dependencies in a group in the shape of a
        cycle.
        Ordering: Order in which Rots appear in the RotComp.
        [swapping/moving] Over: operation that might change a Rot's value in order to preserve the RotComp's. As opposed
        to under, in which case the Rot will keep its value while the one it went under might change its.
        Order: Index of a Rot inside the RotComp, ie its place in the sequence of Rots.
        Id: Unique identifier for each Rot in the RotComp. This moves with the identified Rot, so that it can be
        tracked.
        src_rot, dst_rot: Rots respectively at src_order, dst_order.
    """

    def __init__(self, rots=None, *, ids=None, max_index=0):
        """Construct an object representing the composition of these Rots.

        Args:
            rots: (Optional) Rot or sequence of Rots. By default, interpreted as an empty sequence.
            ids: (optional, keyword-only) Unique identifiers for those Rots.
            max_index: (optional, keyword-only) Maximum index expected to be found inside the Rots of this sequence.

        Raises:
            RotCompError: if rots cannot be parsed as a Rot nor a sequence of Rots.
        """
        if not rots:
            super().__init__([])
        else:
            try:
                first_rot = rots[0]
            except IndexError:
                pass

            except TypeError:
                raise RotCompError

            else:
                try:
                    iter(first_rot)
                except TypeError:
                    rots = [rots]

            super().__init__([Rot(rot) for rot in rots])

        if ids is None:
            if isinstance(rots, RotComp):
                self._ids = rots._ids
            else:
                self.reset_ids()
        else:
            if len(set(ids)) < len(ids):
                raise RotCompIdsError

            self._ids = ids[:len(self)]
            min_available_id = self._min_available_id
            self._ids += list(range(min_available_id, min_available_id + len(self) - len(self._ids)))

        if isinstance(rots, RotComp):
            self._max_index = rots.max_index
        else:
            self._max_index = max_index

    @classmethod
    def from_random(cls, n_rots=1, max_n_rots=None, *, max_index=16, len_=2, max_len=None):
        """Alternate constructor, generate a random RotComp.

        Args:
            n_rots: (optional) Number of Rots to generate. By default, 1.
            max_n_rots: (optional) Maximum number of Rots to generate, making their number random. By default, the
            number is fixed to n_rots.
            max_index: (optional, keyword-only) Maximum index to be found in these Rots. By default, 16.
            len_: (optional, keyword-only) Number of indices per Rot. By default, 2.
            max_len: (optional, keyword-only) Maximum number of indices per Rot, making their number random. By default,
            the number is fixed to len_.
        """
        if max_n_rots is not None:
            n_rots = random.randint(n_rots, max_n_rots)
        return cls([Rot.from_random(max_index, len_, max_len) for _ in range(n_rots)])

    def compress(self):
        """Compress the representation of self without changing its value. """
        self[:] = self.compressed()

    def compressed(self):
        """Return an equal-valued compressed representation of self. """
        from linear_puzzle import LinearPuzzle
        src_perm = LinearPuzzle.from_rotcomp(self)
        dst_perm = LinearPuzzle(src_perm)
        dst_perm.rot(self)
        src_perm.define_solved_perm(dst_perm)
        return src_perm.get_rotcomp_solution()

    def to_bis(self, order=None, *, use_ids=False):
        """Change the representation of self to one that's made out of bis, by way of subdivision.

        Args:
            order: (optional) The order of the Rot to subdivide. By default, subdivides all the Rots.
            use_ids: (optional, keyword-only) If True, order is interpreted as an id. By default, False.
        """
        self._subdivide(2, order, use_ids=use_ids)

    def to_tris(self, order=None, *, use_ids=False, be_strict=False):
        """Change the representation of self to one that's made out of tris, by way of subdivision and growth.

        Args:
            order: (optional) The order of the Rot to subdivide. By default, subdivides and grows all the Rots.
            use_ids: (optional, keyword-only) If True, order is interpreted as an id. By default, False.
            be_strict: (optional, keyword-only) If True, raises a RotCompSubdivideError if some rots can't be made
            into tris. By default, False.

        Raises:
            RotCompSubdivideError: if be_strict and some Rots couldn't be made into tris.
        """
        self._subdivide(3, order, use_ids=use_ids, be_strict=be_strict)

    def randomize_ordering(self):
        """Randomize the ordering of the Rots in self, while preserving its value. """
        dst_ordering = list(self._ids)
        random.shuffle(dst_ordering)
        self._change_ordering(dst_ordering)

    def grow(self, dst_order=0, amount=1, *, use_ids=False):
        """Make this Rot and the following longer, while preserving self's value.

        Args:
            dst_order: (optional) The order of the Rot to grow, along with the following Rot. By default, 0.
            amount: (optional) The amount by which to grow the pair of Rots. By default, 1.
            use_ids: (optional, keyword-only) If True dst_rot is interpreted as an id. By default, False.

        Returns:
            RotCompGrowError: if the requested growth couldn't be achieved.
        """
        if amount == 0:
            return
        elif amount < 0:
            raise RotCompGrowError("amount has to be a positive integer. ")

        if use_ids:
            dst_order = self._get_order_from_id(dst_order)

        src_order = dst_order + 1

        src_rot: Rot
        dst_rot: Rot
        src_rot, dst_rot = self[src_order], self[dst_order]  # Might raise IndexError, intentionally not caught.

        common_indices = self._get_common_indices(dst_order, src_order)

        for dst_index in dst_rot:
            if dst_index not in common_indices:
                break

        else:
            raise RotCompGrowError("Can't grow, because dst_rot doesn't have an index it doesn't share with src_rot. ")

        for src_index in src_rot:
            if src_index not in common_indices:
                break

        else:
            raise RotCompGrowError("Can't grow, because src_rot doesn't have an index it doesn't share with dst_rot. ")

        dst_rot.roll_to(dst_index)
        src_rot.roll_to(src_index, to_front=False)

        unusable_indices = set(src_rot + dst_rot)
        virtual_rot = Rot([dst_index])
        for _ in range(1, amount):
            for middle_index in range(self.max_index + 1):
                if middle_index not in unusable_indices:
                    break

            else:
                raise RotCompGrowError("Out of available indices to use for growing. ")

            virtual_rot.append(middle_index)
            unusable_indices.add(middle_index)
        virtual_rot.append(src_index)

        self.insert(src_order, virtual_rot)
        src_order += 1
        self.insert(src_order, -virtual_rot)
        src_order += 1

        self.fuse(dst_order, dst_order + 1)
        src_order -= 1
        self.fuse(src_order, src_order - 1)

    def fuse(self, dst_order=0, *src_orders, use_ids=False):
        """Fuse Rots together while preserving self's value, and canceling out relevant Rots.

        If two Rots cancel out, stop fusing.

        Args:
            dst_order: (optional) The order of the Rot to fuse into. By default, 0.
            *src_orders: The orders of rots to fuse into dst_order. By default, repeatedly fuse the rot
            following dst_order until no more fuse is possible.
            use_ids: (optional, keyword-only) If True, orders are interpreted as ids. By default, False.

        Raises:
            RotCompFuseError: if src_orders are passed explicitly and they can't be fused into dst_order.
        """
        if use_ids:
            dst_order = self._get_order_from_id(dst_order)
            src_orders = [self._get_order_from_id(src_order) for src_order in src_orders]

        do_raise = True

        if not src_orders:
            src_orders = count(dst_order+1)
            do_raise = False

        src_rot: Rot
        dst_rot: Rot

        for n_fused, src_order in enumerate(src_orders):
            src_order -= n_fused
            if dst_order == src_order:
                continue

            try:
                dst_rot, src_rot = self[dst_order], self[src_order]
            except IndexError as e:
                if do_raise:
                    raise e

                break

            cancel_out = dst_rot == -src_rot
            if cancel_out:
                if dst_order < src_order:
                    self.move_back(src_order, dst_order + 1)
                else:
                    self.move_back(src_order, dst_order - 1)
                    dst_order, src_order = src_order, dst_order
                del self[dst_order:dst_order + 2]
                break

            common_indices = self._get_common_indices(dst_order, src_order)
            if len(common_indices) != 1:
                if do_raise:
                    raise RotCompFuseError

                break

            common_index = common_indices.pop()

            if dst_order < src_order:
                self.move_back(src_order, dst_order + 1)
            else:
                self.move_back(src_order, dst_order - 1)
                dst_order, src_order = src_order, dst_order
                dst_rot, src_rot = src_rot, dst_rot

            dst_rot.roll_to(common_index)
            src_rot.roll_to(common_index, to_front=False)
            dst_rot += src_rot[1:]

            del self[dst_order + 1]

    def move_back(self, src_order, dst_order=0, *, use_ids=False):
        """Move Rot at src_order under to dst_order, while preserving self's value.

        This is a convenience method, equivalent to self.move called with is_back=True. Doing this, src_rot's value
        won't change but that of the Rots it moves under might, as needed.

        Args:
            src_order: Order of the Rot to move.
            dst_order: (optional) Order to move it to. By default, 0.
            use_ids: (optional, keyword-only) If True, orders are interpreted as ids. By default, False.
        """
        self.move(src_order, dst_order, is_back=True, use_ids=use_ids)

    def move(self, src_order, dst_order=0, *, is_back=False, use_ids=False):
        """Move Rot at src_order to dst_order, while preserving self's value.

        Args:
            src_order: Order of the Rot to move.
            dst_order: (optional) Order to move it to. By default, 0.
            is_back: (optional, keyword-only) If True, move src_rot under the rots it moves past, ie without changing
            its value but by changing theirs. By default, False, ie change the value of src_rot as needed to maintain
            self's value.
            use_ids: (optional, keyword-only) If True, orders are interpreted as ids. By default, False.
        """
        if use_ids:
            src_order = self._get_order_from_id(src_order)
            dst_order = self._get_order_from_id(dst_order)

        n_steps = abs(dst_order - src_order)
        if n_steps == 0:
            return

        order_shift = (dst_order - src_order) // n_steps
        for current_order in range(src_order, dst_order, order_shift):
            self._swap(current_order, current_order + order_shift, is_back=is_back)

    def reset_ids(self):
        """Set Rots' ids to be equal to their orders. """
        self._ids = list(range(len(self)))

    def print_with_orders(self, *, use_ids=False):
        """Print the RotComp along with the orders of the Rots on the line beneath, nicely aligned. """
        str_rotcomp = repr(self)
        str_before_list = f"{type(self).__name__}(["
        len_before_list = len(str_before_list)
        str_orders = " " * (len_before_list + 1)
        str_list = str_rotcomp[len_before_list:]
        for order, str_rot in enumerate(str_list.split("[")[1:]):
            str_order = str(self._get_id_from_order(order) if use_ids else order)
            str_orders += str_order + " " * (len(str_rot) - len(str_order) + 1)
        print(str_rotcomp)
        print(str_orders)

    def print_with_ids(self):
        """Print the RotComp along with the ids of the Rots on the line beneath, nicely aligned. """
        self.print_with_orders(use_ids=True)

    def count_by_len(self, len_):
        """Return the number of Rots of the requested len that are present in self. """
        return sum(1 if len(rot) == len_ else 0 for rot in self)

    def append(self, rot):
        super().append(Rot(rot))
        self._ids.append(self._min_available_id)

    def insert(self, order, rot):
        super().insert(order, Rot(rot))
        self._ids.insert(order, self._min_available_id)

    @property
    def max_index(self):
        """Maximum index found or expected to be found in self. """
        max_index = 0
        for rot in self:
            max_index = max(*rot, max_index)
        self._max_index = max(self._max_index, max_index)
        return self._max_index

    def _subdivide(self, len_, order=None, *, use_ids=False, be_strict=False):
        """Attempt to subdivide and grow the Rots in self into the requested len. Non-public method.

        Args:
            len_: Expected len of the Rots.
            order: (optional) The order of the Rot to affect. By default, subdivides and grows all the Rots.
            use_ids: (optional, keyword-only) If True, order is interpreted as an id. By default, False.
            be_strict: (optional, keyword-only) If True, raises a RotCompSubdivideError if some rots can't be made
            into the requested len. By default, False.

        Raises:
            RotCompSubdivideError: if be_strict and some Rots couldn't be made into the requested len.
        """
        if use_ids:
            order = self._get_order_from_id(order)

        new_rotcomp = type(self)()
        new_ids = []
        min_available_id = self._min_available_id

        for order_, (rot, id_) in enumerate(zip(self, self._ids)):
            new_ids.append(id_)
            if order is None or order_ == order:
                subdivs = rot.subdivided(len_)
                new_rotcomp += subdivs
                for _ in range(len(subdivs) - 1):
                    new_ids.append(min_available_id)
                    min_available_id += 1
            else:
                new_rotcomp.append(rot)
        self[:] = new_rotcomp[:]

        self._ids = new_ids

        if order is not None:
            return

        self._sort_rots_by_len(reverse=True)

        self._grow_rots_to(len_, be_strict=be_strict)

    def _grow_rots_to(self, len_, *, be_strict=False):
        """Attempt to grow all the Rots in self up to the requested len. Non-public method.

        Args:
            len_: Expected len to grow the Rots up to.
            be_strict: (optional, keyword-only) Raise an exception if some Rots couldn't be grown as requested.

        Raises:
            RotCompSubdivideError: if some Rots couldn't be grown to the requested len.
        """
        n_rots = len(self)
        n_rots_visited = 0
        for len_to_grow in range(2, len_):
            n_rots_len = self.count_by_len(len_to_grow)
            if be_strict and n_rots_len % 2 != 0:
                raise RotCompSubdivideError
            n_rots_grown = 0
            while n_rots_len - n_rots_grown >= 2:
                self.grow(n_rots - n_rots_len - n_rots_visited + n_rots_grown, len_ - len_to_grow)
                n_rots_grown += 2
            n_rots_visited += n_rots_len

    def _sort_rots_by_len(self, *, reverse=False):
        """Sort the Rots in self by len. Non-public method.

        Args:
            reverse: (optional) If True, sort them in reverse ordering. By default, False.
        """
        ids_and_lens = [(id_, len(rot)) for id_, rot in zip(self._ids, self)]
        ids_and_lens.sort(key=lambda id_and_len: id_and_len[1], reverse=reverse)
        self._change_ordering([id_ for id_, _ in ids_and_lens])

    def _change_ordering(self, dst_ordering):
        """Change self's ordering to this. Non-public-method.

        Args:
            dst_ordering: Requested new ordering, expressed in terms of ids.
        """
        for dst_order, src_id in enumerate(dst_ordering):
            self.move(self._get_order_from_id(src_id), dst_order)

    def _swap(self, src_order, dst_order, *, is_back=False):
        """Swap two adjacent Rots. Non-public method.

        Args:
            src_order: Rot to swap over.
            dst_order: Rot to swap under.
            is_back: If True, reverse the swapping: over becomes under, and conversely. By default, False.

        Raises:
            RotCompSwapError: if src_rot and dst_rot aren't adjacent.
        """
        if is_back:
            src_order, dst_order = dst_order, src_order

        dir_ = dst_order - src_order

        if abs(dir_) > 1:
            raise RotCompSwapError

        if self[src_order] == self[dst_order] or not dir_:
            return

        ted_rot = self._remapped_through(src_order, dst_order)
        self[dst_order], self[src_order] = ted_rot, self[dst_order]

        self._ids[dst_order], self._ids[src_order] = self._ids[src_order], self._ids[dst_order]

    def _remapped_through(self, src_order, dst_order):
        """Return the new value of a Rot swapping over. Non-public method. """
        dir_ = dst_order - src_order

        src_rot, dst_rot = self[src_order], self[dst_order]
        if src_rot == dst_rot:
            return src_rot

        remapped_rot = Rot(src_rot)
        for i, index_ in enumerate(dst_rot):
            if index_ not in src_rot:
                continue

            remapped_rot[src_rot.index(index_)] = dst_rot[(i-dir_) % len(dst_rot)]

        return remapped_rot

    def _roll_rot_at(self, order, roll_amount=1):
        """Roll the Rot found at the requested order. Non-public method. """
        self[order][:] = self[order].rolled(roll_amount)

    def _compress_old(self):
        """Unfinished implementation, do not use as is.

        Attempting to compress a RotComp by analysing its Rots, their dependencies and cycles. Couldn't find a way to
        untangle cycles reliably. The new implementation of compress relies on observing the result of the RotComp as
        applied to a LinearPuzzle, since observing it is sufficient to deduce what the RotComp has done and to
        re-encode it on the fly.
        """
        self.to_bis()
        self.reset_ids()

        print("\n--> Compress starts. ")  # for debug
        self.print_with_ids()  # for debug

        groups, cycles = self._get_groups_and_cycles()
        print("cycle 0:", cycles[0][0])  # for debug

        for group, group_cycles in zip(groups, cycles):
            for cycle in group_cycles:
                if not cycle:
                    continue

                print("\n--> Moving cycle at the beginning of RotComp. ")  # for debug
                for dst_order, id_ in enumerate(sorted(cycle)):
                    self.move_back(
                        self._get_order_from_id(id_),
                        dst_order,
                        )
                    self.print_with_ids()  # for debug

                print("\n--> Placing rots in the order of cycle. ")  # for debug
                for dst_order, id_ in enumerate(cycle):
                    self.move(
                        self._get_order_from_id(id_),
                        dst_order,
                        )
                    self.print_with_ids()  # for debug
                print("\n--> Fusing and separating to make rots continuous. ")  # for debug
                self.fuse()
                self.print_with_ids()  # for debug
                self.to_bis(0)
                self.print_with_ids()  # for debug
                print("\n--> Closing cycle. ")  # for debug
                self.move(
                    dst_order,
                    1,
                    )
                self.print_with_ids()  # for debug
                print("\n--> Canceling out. ")  # for debug
                self.fuse()  # Canceling out
                self.print_with_ids()  # for debug
                print("\n--> Fusing the rest of the cycle. ")  # for debug
                self.fuse()  # Fusing the rest of the cycle
                self.print_with_ids()  # for debug
                break  # for debug

            break  # for debug

    def _get_groups_and_cycles(self):
        """Return the groups and respective cycles of self. Non-public method.

        Raises:
             RotCompError: if self is not made out of bis.
        """
        if any(len(rot) != 2 for rot in self):
            raise RotCompError("RotCompo._get_groups_and_cycles only works on RotComps made out of bis. ")

        groups = []
        cycles = []
        visited_indices = set()
        for index in self._sorted_indices:
            if index in visited_indices:
                continue
            group, group_cycles = self._analyse_dependencies(orders=[], indices=[index])
            for order in group:
                visited_indices.update(self[order])
            groups.append(group)
            cycles.append(group_cycles)

        return groups, cycles

    def _analyse_dependencies(self, orders, indices):
        """Recursive part of self._get_groups_and_cycles. Non-public method.

        Only works on RotComps made out of bis.

        Args:
            orders: Chain of orders making up a group.
            indices: Respectively to orders, indices in common between the Rots in orders.

        Returns:
            group: List of orders that make up the group that was found.
            group_cycles: List of lists of orders making up the cycles in group.
        """
        group = []
        group_cycles = []

        active_index = indices[-1]

        for order, rot in enumerate(self):
            if order in orders or any(order in cycle for cycle in group_cycles):
                continue

            for roll in rot.all_rolls:
                if active_index == roll[0]:
                    break
            else:
                continue

            group.append(order)
            new_index = roll[-1]

            new_orders = orders + [order]
            new_indices = indices + [new_index]

            found_cycle = new_index in indices
            if found_cycle:
                group_cycles.append(new_orders[indices.index(new_index):])
                continue

            desc_group, desc_group_cycles = self._analyse_dependencies(new_orders, new_indices)

            group += desc_group
            group_cycles += desc_group_cycles

        return group, group_cycles

    def _get_common_indices(self, *orders):
        """Return the common indices in all the Rots in self or in the rots designated. Non-public method.

        Args:
            *orders: The orders of Rots to find common indices amongst. By default, find common indices amongst all the
            Rots of self.

        Returns:
            common_indices.
        """
        if orders:
            return set.intersection(*(set(self[order]) for order in orders))

        return set.intersection(*(set(rot) for rot in self))

    def _get_order_from_id(self, id_):
        """Return the order of the Rot identified by id_. """
        if id_ is None:
            return None
        return self._ids.index(id_)

    def _get_id_from_order(self, order):
        """Return the id identifying the Rot at order. """
        if order is None:
            return None
        return self._ids[order]

    @property
    def _sorted_indices(self):
        """The list of unique indices present in the Rots of self, in increasing order. """
        indices = set()
        for rot in self:
            indices.update(rot)
        return sorted(indices)

    @property
    def _min_available_id(self):
        """The minimum id not yet attributed to a Rot in self. """
        if not self._ids:
            return 0

        return max(self._ids) + 1

    def __eq__(self, other):
        return super(type(self), self.compressed()).__eq__(other.compressed())

    def __neg__(self):
        rotcomp = type(self)()
        for rot in reversed(self):
            rotcomp.append(-rot)

        return rotcomp

    def __add__(self, other):
        other = type(self)(other)

        return type(self)(
            super().__add__(other),
            ids=self._ids + [id_ + self._min_available_id for id_ in other._ids],
            max_index=max(self.max_index, other.max_index)
            )

    def __iadd__(self, other):
        other = type(self)(other)

        super().__iadd__(other)
        self._ids += [id_ + self._min_available_id for id_ in other._ids]
        self._max_index = max(self.max_index, other.max_index)
        return self

    def __sub__(self, other):
        return self + -other

    def __isub__(self, other):
        return self.__iadd__(-other)

    def __str__(self):
        return self.__repr__(with_meta=False)

    def __repr__(self, *, with_meta=True):
        str_meta = f", ids={self._ids}, max_index={self.max_index}" if with_meta else ""
        return f"{type(self).__name__}([{', '.join(repr(list(index_)) for index_ in self)}]{str_meta})"

    def __delitem__(self, order):
        super().__delitem__(order)
        del self._ids[order]

    def __getitem__(self, order):
        super_rtn = super().__getitem__(order)
        if isinstance(order, slice):
            new_rot = type(self)(super_rtn)
            new_rot._ids = self._ids[order]
            return new_rot

        return super_rtn


class Rot(list):
    """Represents a rotation, which is a generalization of the concept of swapping two entities, but to arbitrary
    amounts. They are mainly used inside RotComp objects, and encode a transformation of Puzzle objects. Their
    representation is composed of indices in a sequence, and conceptually on application of that transformation, the
    entity identified by each index will be swapped for the one identified by the preceding index in sequence,
    while the first one is swapped for the last.

    Glossary:
        Roll: alternate, equal-valued representation of the Rot, obtained by rolling it.
        Bi: Rot containing two indices.
        Tri: Rot containing three indices.
        Index: Identifiers that make up the Rot.
    """

    def __init__(self, indices=None):
        """Construct a Rot object.

        Args:
            indices: (optional) The sequence of indices making up this Rot. By default, interpreted as an empty
            sequence.

        Raises:
            RotError: if there's an index that is repeated inside of indices.
        """
        if indices is None:
            super().__init__([])
            return

        if len(set(indices)) != len(indices):
            raise RotError

        super().__init__(indices)

    @classmethod
    def from_random(cls, max_index=16, len_=2, max_len=None):
        """Alternate constructor, generate a random Rot.

        Args:
            max_index: (optional) Maximum index to be found in these Rots. By default, 16.
            len_: (optional) Number of indices per Rot. By default, 2.
            max_len: (optional) Maximum number of indices per Rot, making their number random. By default,
            the number is fixed to len_.
        """
        if max_len is not None:
            max_len = min(max_len, max_index)
            len_ = random.randint(len_, max_len)
        rot = []
        while len(rot) < len_:
            index_ = random.randint(0, max_index - 1)
            if index_ in rot:
                continue

            rot.append(index_)

        return cls(rot)

    def subdivided(self, len_):
        """Return a RotComp representing self subdivided into Rots of the requested len, while preserving its value. """
        subdivs = RotComp()
        for i in range(0, len(self), len_-1):
            indices = self[i:i + len_]
            if len(indices) == 1:
                break

            subdivs.append(type(self)(indices))

        return subdivs

    @property
    def all_rolls(self):
        """Generator for all possible rolls of self. """
        roll = Rot(self)
        for _ in range(len(self)):
            yield roll
            roll.append(roll.pop(0))
            roll = Rot(roll)

    def roll_to(self, index_, *, to_front=True):
        """Roll self until the requested index is at its front.

        Args:
            index_: The index to have in front.
            to_front: (optional) If True, the requested index is brought to the front of self, otherwise to the back. By
            default, True.
        """
        if to_front:
            self[:] = self.rolled(len(self) - 1 - self.index(index_))
        else:
            self[:] = self.rolled(-self.index(index_))

    def rolled(self, roll_amount=1):
        """Return a rolled representation of self.

        Args:
            roll_amount: (optional) The amount to roll it by. By default, 1.
        """
        roll = Rot(self)
        for _ in range(-roll_amount % len(self)):
            roll.append(roll.pop(0))
        return Rot(roll)

    def __neg__(self):
        return type(self)(list(reversed(self)))

    def __eq__(self, other):
        if len(self) != len(other):
            return False

        other = type(self)(other)

        for other_roll in other.all_rolls:
            if super().__eq__(other_roll):
                return True

        return False

    def __repr__(self):
        return f"{type(self).__name__}([{', '.join(repr(index_) for index_ in self)}])"

    def __getitem__(self, key):
        super_rtn = super().__getitem__(key)
        if isinstance(key, slice):
            return type(self)(super_rtn)

        return super_rtn


class RotCompError(Exception):
    pass


class RotCompIdsError(RotCompError):
    def __init__(self, message="There are duplicates in the given ids, when they must be unique. "):
        super().__init__(message)


class RotCompSwapError(RotCompError):
    def __init__(self, message="Can't swap two rots that aren't immediately consecutive. "):
        super().__init__(message)


class RotCompSubdivideError(RotCompError):
    def __init__(self, message=(
            "Couldn't subdivide strictly to the requested len. "
            )):
        super().__init__(message)


class RotCompGrowError(RotCompError):
    pass


class RotCompFuseError(RotCompError):
    def __init__(self, message=(
            "Can't fuse two rots that don't either have exactly one index in common or cancel each other out. "
            )):
        super().__init__(message)


class RotError(Exception):
    def __init__(self, message="Invalid Rot. "):
        super().__init__(message)
