Lists
-----
A list is a finite, ordered sequence of values.
Everything in Curv that is an ordered sequence of values, is also a list,
and all the list operations describe here will work on it.

There are multiple syntaxes for constructing lists:

* ``[1,2,3]`` is a list of 3 numbers.
* ``"abc"`` is a list of 3 characters,
  equivalent to ``[char 97,char 98,char 99]``.
  See `Strings`_ for more about string syntax.
* ``2..7`` is a range, the consecutive integers from ``2`` to ``7``.
  It is equivalent to ``[2,3,4,5,6,7]``.

.. _`Strings`: Strings.rst

List Constructors
  A list constructor is a comma-separated list of expressions or value
  generators, inside square brackets.

  For example::

    []
    [1]
    [1,2]

  A value generator is a kind of `statement`_
  that adds zero or more elements to the list being constructed.
  The simplest value generator is just an expression,
  which adds one element to the list.

  The spread operator (``...list_expression``) interpolates all of the elements
  from some list into the list being constructed.
  For example, this concatenates two lists::

    [...a, ...b]

  Complex value generators may be composed from simpler ones using blocks and control structures,
  as described in `Statements`_.
  This syntax is a generalization of the *list comprehensions* found in other languages.
  For example, this yields ``[1,4,9,16,25]``::

    [for (i in 1..5) i^2]

.. _`statement`: Statements.rst
.. _`Statements`: Statements.rst

Range constructors
  ``i .. j``
    Returns the ascending list of numbers: ``i``, ``i+1``, ``i+2``, ...
    up to ``j`` inclusive.
    For example, ``1..10`` is ``[1,2,3,4,5,6,7,8,9,10]``.
    Same as ``i .. j by 1``.

  ``i .. j by k``
    Same as ``i..j``, except that the step value is ``k`` instead of ``1``.
    The step value may be positive or negative, and need not be an integer.
    For example, ``1..0 by -0.25`` is ``[1, 0.75, 0.5, 0.25, 0]``.

    We assume that ``j-i`` is intended to be an integer multiple of ``k``.
    It might not be an exact multiple, due to floating point inaccuracy,
    so the number of list elements is computed using rounding, like this::

      let n = round((last - first)/step) in
      if (n < 0) 0 else n + 1
    
    Using this algorithm, the number of list elements will always be
    correct (under the stated assumption), but the final list element might
    be larger or smaller than expected, due to floating point inaccuracy.
    For example, ``0.1 .. 0.7 by 0.2`` returns::

      [0.1, 0.30000000000000004, 0.5, 0.7000000000000001]

    Some languages have an integer range function that guarantees not to
    generate values greater than the end value, regardless of the step value.
    But if you transfer this guarantee to floating point ranges, then you get
    buggy behaviour, as the above example shows. Integer and floating point
    stepped ranges are distinct concepts, which should have separate names to
    avoid confusion. If stepped integer range semantics are required in Curv,
    they must be implemented as a separate function.
    Here is a sample implementation of a Python-like integer range::

      irange{lo=0,hi,step=1} = [local n=lo; while (n<hi) (n; n:=n+step)]

  ``i ..< j``
    Same as ``i..j`` except that the final element in the sequence is omitted.
  
  ``i ..< j by k``
    Same as ``i..j by k`` except that the final element in the sequence is omitted.

List Indexing
  ``a.[i]``
    The i'th element of list ``a``, if ``i`` is an integer.
    Zero based indexing: ``a.[0]`` is the first element.

    Note: the syntax ``a.[i]`` is used for list indexing, rather than the
    more popular ``a[i]`` syntax, because in Curv, ``a[i]`` means
    a call to the function ``a`` with argument ``[i]``.

    For multi-dimensional array indexing, see `Arrays`_.

List Slicing
  ``a.[indices]``
    Returns ``[a.[indices.[0]], a.[indices.[1]], ...]``,
    where ``indices`` is a list of integers.
    For example, ``a.[0..<3]`` returns a list of the first 3 elements of ``a``.

    For multi-dimensional array slicing, see `Arrays`_.

.. _`Arrays`: Arrays.rst

``count a``
  The number of elements in list ``a``.

``is_list value``
  True if the value is a list, false otherwise.

``a ++ b``
  Infix binary list concatenation.

  * ``[1,2] ++ [3,4]`` returns ``[1,2,3,4]``.
  * ``"ab" ++ "cd"`` returns ``"abcd"``.
  * ``"ab" ++ [3,4]`` returns ``"ab"++[3,4]``.
    We use this syntax for printing a list containing a mixture of character
    and non-character elements because it's easier to read than the
    alternatives.

``concat aa``
  This is the general list concatenation operator.
  ``aa`` is a list of lists. The component lists are catenated.
  For example, ``concat[[1,2],[3,4]]`` is ``[1,2,3,4]``.
  ``concat[]`` is ``[]``, because ``[]`` is the
  identity element for the concatenation operation.

``reverse a``
  A list containing the elements of ``a`` in reverse order.

``map f a``
  The list obtained by applying ``f`` to each element of list ``a``.
  Equivalent to ``[for (x in a) f x]``.

``filter p a``
  The list of those elements of ``a`` that satisfy the predicate ``p``.
  Equivalent to ``[for (x in a) if (p x) x]``.

``reduce [zero, f] a``
  Using binary function ``f``,
  iteratively combine all of the elements of list ``a`` into a single value,
  recursing on the left.
  For a 4 element list ``[a,b,c,d]``, this will compute::

    f[f[f[a,b],c],d]

  If the list has zero length, the result is ``zero``.

``contains [list, x]``
  A predicate, returns ``#true`` if ``x`` is an element of ``list``.
